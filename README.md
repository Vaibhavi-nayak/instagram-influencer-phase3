# Phase 3 – Instagram Influencer Score Predictor (Streamlit)

## Run
```
pip install -r requirements.txt
streamlit run app.py
```
Keep `app.py`, `best_model_pipeline.joblib`, the CSV files and the `plots/` folder together.

## Where the influencer data comes from
| Case | Data used |
|---|---|
| Influencer is in the dataset (200 top + 5 Indian test accounts) | **Exact stored values** from the CSV files |
| Any other Instagram ID | Button **Fetch live stats** tries, in order: official Graph API (if token set in sidebar) -> **instaloader** (`instagram_fetch.py`, no login) -> Instagram web profile JSON (followers, posts, likes of last ~12 posts -> avg likes & engagement) -> page meta tags (followers/posts only). Anything not obtainable is pre-filled with labelled estimates you can overwrite. Private accounts cannot be fetched |

No random or synthetic profiles are generated any more.

## Graphs shown for every selected influencer
1. Score position (histogram of all 205 scores, predicted/actual marked)
2. Metric percentiles (6 metrics vs all influencers)
3. Followers-vs-score and engagement-vs-score scatter plots (influencer highlighted)
4. "Why this score" – sensitivity of the prediction to each feature
5. Five most similar influencers

Same graphs are available for the 5-influencer benchmark and for any row of an uploaded CSV.
The evaluation page also has a per-influencer actual-vs-predicted check for the Training, Validation and External splits.
