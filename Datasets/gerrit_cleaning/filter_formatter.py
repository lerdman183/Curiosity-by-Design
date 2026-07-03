import os
import pandas as pd


def filterQuestionsAndSave():
    folder = os.path.dirname(__file__)
    csv_path = os.path.join(folder, 'cleaned_commits_large_with_line_number_with_id.csv')
    out_path = os.path.join(folder, 'filtered_questions.json')

    df = pd.read_csv(csv_path, dtype=str)

    # Determine columns: prefer named columns 'comment' and 'before',
    # otherwise fallback to the 6th and 7th columns (0-based indices 5 and 6)
    if {'comment', 'before'}.issubset(df.columns):
        cols = ['comment', 'before']
    else:
        # ensure there are enough columns
        if df.shape[1] >= 7:
            cols = [df.columns[5], df.columns[6]]
        else:
            raise SystemExit('CSV does not contain expected columns')

    sub = df.loc[:, cols].astype(str)

    # Keep rows where either column contains a question mark
    mask = sub[cols[0]].str.contains('\?', na=False)
    filtered = sub[mask]

    # Rename and reorder columns for output
    filtered = filtered.rename(columns={
    "comment": "clarifying_question",
    "before": "prompt"})

    filtered = filtered[["prompt", "clarifying_question"]]

    filtered.to_json(out_path, orient='records', force_ascii=False)

filterQuestionsAndSave()