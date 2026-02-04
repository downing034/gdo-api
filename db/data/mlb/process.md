MLB Data Processing Workflow

1. Download Current Season Leader Reports from FanGraphs
	•	Navigate to the FanGraphs season leaders page.
	•	Ensure you’re logged into your account to access saved reports.
	•	Click “Load / Save Report”.
	•	Load and export the following reports (all prefixed with 0-):
	•	0-current-pitchers
	•	0-current-batters
	•	0-current-batters-vs-lhp
	•	0-current-batters-vs-rhp
	•	0-team-pitching
	•	0-team-batting

2. Download Park Factors
	•	Navigate to the FanGraphs Guts page.
	•	Export the park factor table as a CSV.
	•	Rename the downloaded file to park-factors.csv.

3. Standardize and Upload Files
	•	When exported, reports automatically drop the 0- prefix (e.g., 0-current-batters.csv becomes current-batters.csv).
	•	Upload all exported files to the appropriate project directory.
	•	If using a Flask API, restart the API server to load the new data.

4. Handle Name Mismatches
	•	There are custom player name mappings required to normalize name differences between FanGraphs CSV exports and MLB StatsAPI.
	•	These mappings are handled in:
	•	Frontend: prediction-helpers.ts
	•	Backend: player_data.py
	•	These mappings should be reviewed and updated periodically as player names or exports change.

5. Model Training Caveats
	•	Sometimes season-long leader data in the current CSVs differs from historical box score data from FanGraphs.
	•	This discrepancy only affects model training and not daily predictions.
	•	These differences should be noted during feature validation to avoid data integrity issues.

6. Model Result Tracking
	•	All model results, including:
	•	Your predictions
	•	Sportsline’s predictions
	•	ESPN’s odds
	•	Are currently tracked in a shared Google Sheet (CSV format).
	•	This process may change in the future, but it is the current method for centralizing result comparisons and evaluations.