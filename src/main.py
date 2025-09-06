"""Support Email NLP - main processing script

This script reads `data/Sample_Support_Emails_Dataset.csv`, performs light NLP
(urgency, topic, sentiment, summary, auto-reply), and writes `output/processed_emails.csv`.

It uses OpenAI if `OPENAI_API_KEY` is set; otherwise falls back to simple heuristics.
"""
import os, csv, json
import pandas as pd
import re

DATA_IN = os.path.join('data', 'Sample_Support_Emails_Dataset.csv')
OUT = os.path.join('output', 'processed_emails.csv')

def clean_text(text):
    if pd.isna(text):
        return ''
    text = str(text)
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.split(r"\n--\n|\nRegards,|\nBest,|\nThanks,", text)[0]
    return text.strip()

def detect_urgency(text):
    t = (text or '').lower()
    if any(k in t for k in ['urgent','critical','immediate','downtime','blocked']):
        return 'high'
    if any(k in t for k in ['help','support','issue','problem','error']):
        return 'medium'
    return 'low'

def detect_topic(text):
    t = (text or '').lower()
    if 'billing' in t or 'refund' in t or 'pricing' in t:
        return 'billing/refund'
    if 'login' in t or 'password' in t or 'account' in t or 'access' in t:
        return 'login/account'
    if 'integration' in t or 'api' in t or 'crm' in t:
        return 'integration/api'
    if 'downtime' in t or 'system' in t or 'outage' in t:
        return 'system/downtime'
    return 'general'

def simple_sentiment(text):
    t = (text or '').lower()
    if any(k in t for k in ['thank','great','appreciate','good','love']):
        return 'positive'
    if any(k in t for k in ['angry','bad','worst','unacceptable','terrible']):
        return 'negative'
    return 'neutral'

def summarize(text, max_words=25):
    words = str(text or '').split()
    return ' '.join(words[:max_words]) + ('...' if len(words) > max_words else '')

def generate_reply(row):
    if row.get('likely_urgency') == 'high':
        return f"Dear {row.get('sender')},\n\nWe understand your issue is urgent regarding {row.get('subject_topic')}. Our team is already working on it and will update you shortly.\n\nBest regards,\nSupport Team"
    if row.get('likely_urgency') == 'medium':
        return f"Dear {row.get('sender')},\n\nThank you for reaching out about {row.get('subject_topic')}. We are reviewing your request and will get back soon.\n\nBest regards,\nSupport Team"
    return f"Dear {row.get('sender')},\n\nThanks for contacting us. We noted your query about {row.get('subject_topic')} and will respond in due course.\n\nBest regards,\nSupport Team"

def try_openai_summary_and_reply(text):
    """Attempt to use OpenAI for summary and reply if API key is set. Returns (summary, reply) or (None,None)."""
    key = os.environ.get('OPENAI_API_KEY')
    if not key:
        return None, None
    try:
        import openai
        openai.api_key = key
        prompt_sum = f"Provide a 1-2 sentence summary of the following support email:\n\n{text}"
        resp = openai.ChatCompletion.create(model=os.getenv('OPENAI_MODEL','gpt-4o-mini'), messages=[{'role':'user','content':prompt_sum}], max_tokens=120, temperature=0.2)
        summary = resp.choices[0].message['content'].strip() if resp and resp.choices else None
        prompt_reply = f"Draft a concise support reply (4-6 sentences) for the email:\n\n{text}"
        resp2 = openai.ChatCompletion.create(model=os.getenv('OPENAI_MODEL','gpt-4o-mini'), messages=[{'role':'user','content':prompt_reply}], max_tokens=250, temperature=0.2)
        reply = resp2.choices[0].message['content'].strip() if resp2 and resp2.choices else None
        return summary, reply
    except Exception as e:
        print('OpenAI call failed, falling back to heuristics:', e)
        return None, None

def main():
    if not os.path.exists(DATA_IN):
        print('Input file not found at', DATA_IN)
        return
    df = pd.read_csv(DATA_IN)
    # ensure columns exist
    for col in ['sender','subject','body']:
        if col not in df.columns:
            df[col] = ''

    df['subject'] = df['subject'].fillna('')
    df['body'] = df['body'].fillna('')
    df['clean_body'] = df['body'].apply(clean_text)
    # urgency: take max of subject/body heuristic
    df['urg_subj'] = df['subject'].apply(detect_urgency)
    df['urg_body'] = df['clean_body'].apply(detect_urgency)
    order = {'low':0,'medium':1,'high':2}
    df['likely_urgency'] = df.apply(lambda r: r['urg_subj'] if order[r['urg_subj']]>=order[r['urg_body']] else r['urg_body'], axis=1)
    df['subject_topic'] = df['subject'].apply(detect_topic)
    df['sentiment'] = df['clean_body'].apply(simple_sentiment)
    # Try OpenAI for better summary/reply, fallback to heuristics
    summaries = []
    replies = []
    for _, row in df.iterrows():
        s, rep = try_openai_summary_and_reply(row['clean_body'])
        if s is None:
            s = summarize(row['clean_body'])
        if rep is None:
            rep = generate_reply({'sender':row['sender'],'subject_topic':row['subject_topic'],'likely_urgency':row['likely_urgency']})
        summaries.append(s)
        replies.append(rep)
    df['summary'] = summaries
    df['auto_reply'] = replies

    os.makedirs('output', exist_ok=True)
    df.to_csv(OUT, index=False)
    print('Processed output saved to', OUT)

if __name__ == '__main__':
    main()
