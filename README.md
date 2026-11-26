# Pee Pee League HQ

Live best-ball dashboard for the Pee Pee Sleeper league (10-team, full PPR).

- **index.html** — the whole site. Static page, no build step.
- **data.json** — weekly model output: player-level Monte Carlo sim (playoff / bye / title odds), toilet-bowl odds, matchup projections, rosters, waiver board. Regenerated every Tuesday by a scheduled Claude task, which commits it here.
- **d3.v7.min.js** — vendored D3 7.9.0.

While the page is open it pulls live scores, game states, and updated projections straight from Sleeper's public API (`api.sleeper.app`), re-fetching every minute during live games. Lineups and team totals are scored best-ball style: best QB, 2 RB, 2 WR, TE, 2 flex over actual points for finished games, max(actual, projected) for live games, and injury-scaled projections for upcoming ones.
