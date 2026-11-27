"""Player-level best-ball season model for the Pee Pee league. Primary data: Sleeper league
box scores weeks 1-11 (players_points), live injury tags, real NFL schedule/byes, real
H2H pairings weeks 12-14, league settings (6 playoff teams, top-2 byes, weeks 15-17).

Pipeline:
1. Per-player weekly mean: exponentially weighted average of played games (recency
   half-life H per position) shrunk toward the position baseline by k pseudo-games;
   H and k are chosen by walk-forward backtest on weeks 7-11 (predict each played
   game from prior weeks only, minimize MAE).
2. Variance: position-level heteroscedastic sd(mu) = a + b*mu fit on backtest residuals.
   Scores are sampled from a Gamma with that mean and sd (right-skew, floor at 0).
3. Forward weekly mean blends model EWMA (65%) with Sleeper's weekly projection (35%)
   where one exists, since the projection carries matchup/injury context the logs lack.
   The blend weight will be refit once weekly projections have been logged alongside
   actuals for a few weeks.
4. Availability: bye weeks are zeros from the real schedule; injury designations map to
   play probabilities (IR .05, Out 0 next week then .5, Doubtful .25 then .8,
   Questionable .78, Suspended 0 then .9, healthy .96, no NFL team .2).
5. Team week = best-ball optimum over sampled player scores (QB/2RB/2WR/TE/2FLEX),
   so bench depth and spike capture are priced by the actual scoring operator.
6. Season: real pairings weeks 12-14, seeding by wins with real points-for carried
   forward as tiebreak, 6-team bracket with top-2 byes, reseeded, weeks 15-17.
"""
import json, math, pathlib, numpy as np

rng = np.random.default_rng(26)
D = pathlib.Path(__file__).parent
league = json.loads((D/'league.json').read_text())
meta   = json.loads((D/'meta.json').read_text())      # id -> [name,pos,team,inj]
proj   = json.loads((D/'proj.json').read_text())      # week -> id -> sleeper ppr proj
weeks  = []
for f in ['wk1-4.json','wk5-8.json','wk9-14.json']:
    weeks += json.loads((D/f).read_text())
hist = {}   # pid -> {week: pts}
for wk in weeks:
    if wk['w'] > 11: continue
    for m in wk['m']:
        for pid, pts in (m.get('pp') or {}).items():
            hist.setdefault(pid, {})[wk['w']] = pts

POS = {'QB','RB','WR','TE'}
pos_of = {pid: (meta[pid][1] if pid in meta and meta[pid][1] in POS else 'WR') for pid in set(list(hist)+ [p for r in league['rosters'] for p in r['players']])}
played = {pid: sorted((w,p) for w,p in h.items() if p > 0) for pid,h in hist.items()}
base = {pos: np.mean([p for pid,g in played.items() if pos_of.get(pid)== pos for _,p in g] or [8.0]) for pos in POS}

def ewma(games, upto, H, k, pos):
    g = [(w,p) for w,p in games if w < upto]
    if not g: return base[pos]*0.75
    wts = [0.5**((upto-1-w)/H) for w,_ in g]
    num = sum(wt*p for wt,( _,p) in zip(wts,g)); den = sum(wts)
    return (num + base[pos]*k*0.6) / (den + k)   # shrink toward 60% of baseline (replacement-ish)

# --- walk-forward fit of (H, k) per position, weeks 7-11 ---
fit, resid = {}, {pos: [] for pos in POS}
for pos in POS:
    best = None
    for H in (2,3,4,6,99):
        for k in (1,2,4):
            errs = []
            for pid,g in played.items():
                if pos_of.get(pid)!=pos: continue
                for w,actual in g:
                    if w < 7 or len([1 for ww,_ in g if ww<w])<2: continue
                    errs.append(abs(ewma(g,w,H,k,pos)-actual))
            mae = np.mean(errs) if errs else 99
            if best is None or mae < best[0]: best = (mae,H,k)
    naive = []
    for pid,g in played.items():
        if pos_of.get(pid)!=pos: continue
        for w,actual in g:
            prior=[p for ww,p in g if ww<w]
            if w>=7 and len(prior)>=2: naive.append(abs(np.mean(prior)-actual))
    fit[pos] = {'mae':round(best[0],2),'H':best[1],'k':best[2],'naive_mae':round(float(np.mean(naive)),2) if naive else None}
    for pid,g in played.items():
        if pos_of.get(pid)!=pos: continue
        for w,actual in g:
            if w<7 or len([1 for ww,_ in g if ww<w])<2: continue
            mu = ewma(g,w,fit[pos]['H'],fit[pos]['k'],pos)
            resid[pos].append((mu, actual-mu))

sdfit = {}
for pos in POS:
    mus = np.array([m for m,_ in resid[pos]]); rs = np.array([r for _,r in resid[pos]])
    A = np.vstack([np.ones_like(mus), mus]).T
    coef, *_ = np.linalg.lstsq(A, np.abs(rs)*math.sqrt(math.pi/2), rcond=None)
    sdfit[pos] = (max(2.0,float(coef[0])), max(0.05,float(coef[1])))

# --- forward means ---
BYES = {13:{'BAL','IND','LV','NYJ'}, 14:{'DAL','ARI'}}
def pplay(pid, w):
    nm, pos, team, inj = meta.get(pid, ['?','WR',None,None])
    if team and w in BYES and team in BYES[w]: return 0.0
    first = (w == 12)
    return {'IR':0.05,'Out':0.0 if first else 0.5,'Doubtful':0.25 if first else 0.8,
            'Questionable':0.78,'Sus':0.0 if first else 0.9}.get(inj, 0.96 if team else 0.2)

def mu_of(pid, w):
    pos = pos_of.get(pid,'WR')
    g = played.get(pid, [])
    m = ewma(g, 12, fit[pos]['H'], fit[pos]['k'], pos)
    sp = proj.get(str(w),{}).get(pid)
    return 0.65*m + 0.35*sp if sp is not None else m

rosters = {r['rid']: r for r in league['rosters']}
owner = {r['rid']: next(u['name'] for u in league['users'] if u['id']==r['owner']) for r in league['rosters']}
pair = {}   # week -> list of (ridA, ridB)
for wk in weeks:
    if wk['w'] in (12,13,14):
        by_mid = {}
        for m in wk['m']: by_mid.setdefault(m['mid'],[]).append(m['rid'])
        pair[wk['w']] = list(by_mid.values())

SIMS = 5000
IDX = {'QB':0,'RB':1,'WR':2,'TE':3}
def team_scores(rid, w, n):
    """n sampled best-ball team scores for roster rid in week w."""
    ps = rosters[rid]['players']
    groups = {'QB':[],'RB':[],'WR':[],'TE':[]}
    for pid in ps: groups[pos_of.get(pid,'WR')].append(pid)
    samp = {}
    for pos, pids in groups.items():
        if not pids: samp[pos]=np.zeros((n,0)); continue
        a,b = sdfit[pos]
        cols=[]
        for pid in pids:
            mu = max(0.5, mu_of(pid,w)); sd = a + b*mu
            shape = (mu/sd)**2; scale = sd*sd/mu
            x = rng.gamma(shape, scale, n)
            x *= (rng.random(n) < pplay(pid,w))
            cols.append(x)
        samp[pos] = np.sort(np.column_stack(cols), axis=1)[:, ::-1]
    qb = samp['QB'][:,0] if samp['QB'].shape[1] else 0
    rb = samp['RB']; wr = samp['WR']; te = samp['TE']
    rb2 = rb[:,:2].sum(1) if rb.shape[1]>=2 else rb.sum(1)
    wr2 = wr[:,:2].sum(1) if wr.shape[1]>=2 else wr.sum(1)
    te1 = te[:,0] if te.shape[1] else np.zeros(n)
    flexpool = np.column_stack([x for x in (rb[:,2:], wr[:,2:], te[:,1:]) if x.shape[1]>0] or [np.zeros((n,1))])
    flex = np.sort(flexpool, axis=1)[:, ::-1][:, :2].sum(1)
    return qb + rb2 + wr2 + te1 + flex

rids = sorted(rosters)
S = {w: {rid: team_scores(rid, w, SIMS) for rid in rids} for w in range(12,18)}
exp_pts = {rid: float(np.mean([S[w][rid].mean() for w in (12,13,14)])) for rid in rids}

wins0 = {rid: rosters[rid]['wins'] for rid in rids}
pf0   = {rid: rosters[rid]['fpts'] for rid in rids}
count = {rid: dict(po=0,bye=0,semi=0,final=0,champ=0,miss=0,waffle_rr=0,waffle_w15=0,last=0) for rid in rids}
for i in range(SIMS):
    wins = dict(wins0); pf = dict(pf0)
    for w in (12,13,14):
        for a,b in pair[w]:
            sa, sb = S[w][a][i], S[w][b][i]
            pf[a]+=sa; pf[b]+=sb
            wins[a if sa>=sb else b]+=1
    order = sorted(rids, key=lambda r:(wins[r], pf[r]), reverse=True)
    seeds, bottom = order[:6], order[6:]
    for r in seeds: count[r]['po']+=1
    for r in seeds[:2]: count[r]['bye']+=1
    for r in bottom: count[r]['miss']+=1
    def game(a,b,w): return a if S[w][a][i]>=S[w][b][i] else b
    w1 = game(seeds[2],seeds[5],15); w2 = game(seeds[3],seeds[4],15)
    for r in (seeds[0],seeds[1],w1,w2): count[r]['semi']+=1
    lo,hi = (w1,w2) if seeds.index(w1)>seeds.index(w2) else (w2,w1)
    f1 = game(seeds[0],lo,16); f2 = game(seeds[1],hi,16)
    for r in (f1,f2): count[r]['final']+=1
    count[game(f1,f2,17)]['champ']+=1
    # Toilet bowl: bottom 4 non-playoff teams; the Waffle House pair under both proposed formats.
    t7,t8,t9,t10 = bottom
    # Option 1: Week 15 alone decides — losers of 7v10 and 8v9 take the punishment.
    l1 = t10 if game(t7,t10,15)==t7 else t7
    l2 = t9  if game(t8,t9,15)==t8 else t8
    for r in (l1,l2): count[r]['waffle_w15']+=1
    # Option 2: weeks 15-17 round robin (7v10/8v9, 7v9/8v10, 7v8/9v10); worst 2 records go,
    # ties broken on points scored in those three weeks.
    rr_w = {r:0 for r in bottom}; rr_pf = {r:0.0 for r in bottom}
    for w,(ga,gb) in zip((15,16,17), (((t7,t10),(t8,t9)), ((t7,t9),(t8,t10)), ((t7,t8),(t9,t10)))):
        for a,b in (ga,gb):
            rr_pf[a]+=S[w][a][i]; rr_pf[b]+=S[w][b][i]
            rr_w[game(a,b,w)]+=1
    rr_order = sorted(bottom, key=lambda r:(rr_w[r], rr_pf[r]))
    for r in rr_order[:2]: count[r]['waffle_rr']+=1
    count[order[-1]]['last']+=1

def pct(c):
    p=100*c/SIMS
    return 0 if c==0 else (0.5 if p<1 else round(p,1))
rows=[]
for rid in sorted(rids, key=lambda r:-count[r]['champ']):
    c=count[rid]
    rows.append({'team':owner[rid],'exp_weekly_pts':round(exp_pts[rid],1),
      'playoffs_pct':pct(c['po']),'bye_pct':pct(c['bye']),'semis_pct':pct(c['semi']),
      'finals_pct':pct(c['final']),'champ_pct':pct(c['champ'])})

toilet=[]
for rid in sorted(rids, key=lambda r:-count[r]['waffle_rr']):
    c=count[rid]
    toilet.append({'team':owner[rid],'miss_playoffs_pct':pct(c['miss']),
      'waffle_rr_pct':pct(c['waffle_rr']),'waffle_wk15_pct':pct(c['waffle_w15']),
      'last_place_pct':pct(c['last'])})
(D/'toilet.json').write_text(json.dumps({'rows':toilet}))
calc_players = {}
for r in rosters.values():
    for pid in r['players']:
        calc_players[pid] = {'m': [round(max(0.5, mu_of(pid, w)), 2) for w in range(13, 18)],
                             'a': [round(pplay(pid, w), 2) for w in range(13, 18)]}
(D/'calc.json').write_text(json.dumps({'weeks': list(range(13, 18)),
  'sd': {p: [round(a, 2), round(b, 3)] for p, (a, b) in sdfit.items()},
  'players': calc_players}))
mymu = {pid: round(mu_of(pid,12),1) for pid in rosters[6]['players']}
matchups = [{'week':12,'teams':[owner[a],owner[b]],
             'win_pct':[round(100*float(np.mean(S[12][a]>=S[12][b])),1),
                        round(100*float(np.mean(S[12][b]>S[12][a])),1)]} for a,b in pair[12]]
out = {'rows':rows,'meta':[
  {'label':'model2.py','value':pathlib.Path(__file__).read_text(),'type':'code','lang':'python'},
  {'label':'Inputs','value':'Sleeper league box scores weeks 1-11 (players_points), live injury tags, real NFL schedule and byes (Wk13 BAL/IND/LV/NYJ, Wk14 DAL/ARI), real H2H pairings weeks 12-14, Sleeper weekly projections weeks 12-17 (35% of the forward mean).'},
  {'label':'Backtest (walk-forward wks 7-11)','value':json.dumps({'fit':fit,'sd_fit':{p:[round(a,2),round(b,3)] for p,(a,b) in sdfit.items()}})},
  {'label':'Assumptions','value':'5,000 player-level season sims; Gamma weekly scores; availability from injury tags (IR .05, Out 0 this wk then .5, Doubtful .25 then .8, Questionable .78, healthy .96); 6-team playoff, top-2 byes, reseeded, weeks 15-17; seeding ties broken on real points-for carried through the sim. Zero odds shown as 0, under 1% as 0.5.'}]}
(D/'odds_model2.json').write_text(json.dumps(out))
(D/'matchup12.json').write_text(json.dumps({'rows':[{'week':m['week'],'team_a':m['teams'][0],'team_b':m['teams'][1],'win_a_pct':m['win_pct'][0],'win_b_pct':m['win_pct'][1]} for m in matchups]}))
print('fit:', json.dumps(fit))
print('sd:', {p:(round(a,1),round(b,2)) for p,(a,b) in sdfit.items()})
for r in rows: print(r)
print('wk12:', matchups)
me = {meta[pid][0]: mymu[pid] for pid in rosters[6]['players']}
print('adamyeet wk12 mu:', dict(sorted(me.items(), key=lambda x:-x[1])))
