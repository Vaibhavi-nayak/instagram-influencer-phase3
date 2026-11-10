"""Instagram Influencer Score Predictor — Phase 3 (Streamlit deployment).

Data policy (what is "accurate" in this app):
  1. Influencers that exist in our dataset (200 top influencers + 5 Indian
     external-test influencers) use their EXACT stored values. Nothing is
     guessed for them.
  2. For any other handle the app tries to read followers / posts / name from
     the public Instagram page (best effort - Instagram often blocks this).
     Instagram does NOT publish avg likes, engagement rate or total likes
     publicly, so those fields are entered by the user (pre-filled with
     clearly labelled estimates). No random / synthetic profiles are used.

Every selected influencer gets its own set of graphs (score position, metric
percentiles, followers/engagement scatter, "why this score" sensitivity chart
and the most similar influencers).
"""

import html
import io
import os
import re
import time

import joblib
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.model_selection import train_test_split

import instagram_fetch

# ------------------------------------------------------------------------------
# Page config & styling
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title='Instagram Influencer Score Predictor',
    page_icon='📸',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown(
    """
<style>
    .main-header {font-size: 2.3rem; color: #1E3A8A; font-weight: 800;
                  margin-bottom: 0.2rem; text-align: center;}
    .sub-header {font-size: 1.05rem; color: #4B5563; text-align: center;
                 margin-bottom: 1.5rem;}
    .metric-card {background-color: #F8FAFC; border-radius: 12px; padding: 1.2rem;
                  border-left: 5px solid #2563EB; box-shadow: 0 2px 6px rgba(0,0,0,0.06);
                  color: #111827;}
    .profile-card {background: linear-gradient(135deg, #EFF6FF 0%, #F5F3FF 100%);
                   border: 1px solid #DBEAFE; border-radius: 12px; padding: 1.1rem;
                   margin-bottom: 1rem;}
    .badge {padding: 4px 12px; border-radius: 16px; font-weight: 700;
            font-size: 0.85rem; display: inline-block;}
    .badge-dataset {background:#DEF7EC; color:#03543F;}
    .badge-live {background:#E1EFFE; color:#1E429F;}
    .badge-manual {background:#FDF6B2; color:#723B10;}
    .badge-tier1 {background:#DEF7EC; color:#03543F; padding:6px 14px; border-radius:20px; font-weight:700;}
    .badge-tier2 {background:#E1EFFE; color:#1E429F; padding:6px 14px; border-radius:20px; font-weight:700;}
    .badge-tier3 {background:#FDF6B2; color:#723B10; padding:6px 14px; border-radius:20px; font-weight:700;}
    .insta-tag {background: linear-gradient(45deg,#f09433 0%,#e6683c 25%,#dc2743 50%,#cc2366 75%,#bc1888 100%);
                color: white; padding: 4px 12px; border-radius: 12px; font-weight: 600;}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# Constants & helpers
# ------------------------------------------------------------------------------
# Streamlit >= 1.50 uses width='stretch'; older versions use use_container_width=True.
_VER = tuple(int(x) for x in re.findall(r'\d+', st.__version__)[:2])
STRETCH = {'width': 'stretch'} if _VER >= (1, 50) else {'use_container_width': True}

SUFFIX = {'k': 1e3, 'm': 1e6, 'b': 1e9}
NUM_COLS = [
    'posts',
    'followers',
    'avg_likes',
    '60_day_eng_rate',
    'new_post_avg_like',
    'total_likes',
]
FEATURE_COLS = NUM_COLS + ['country']
NICE = {
    'posts': 'Posts',
    'followers': 'Followers',
    'avg_likes': 'Avg likes',
    '60_day_eng_rate': '60-day eng. rate',
    'new_post_avg_like': 'New-post avg likes',
    'total_likes': 'Total likes',
    'country': 'Country',
}
COUNTRY_CHOICES = [
    'United States', 'Brazil', 'India', 'Indonesia', 'Spain', 'United Kingdom',
    'Italy', 'South Korea', 'France', 'Canada', 'Turkey', 'Colombia',
    'Netherlands', 'Mexico', 'Argentina', 'Unknown', 'Other',
]
C_PRED, C_ACT, C_REF, C_PEER = '#2563EB', '#10B981', '#CBD5E1', '#F59E0B'


def to_number(value):
    """Converts strings such as 3.3k, 475.8m, 29.0b, 1.39% to float."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower().replace('%', '').replace(',', '')
    if text in ('', 'nan'):
        return np.nan
    if text[-1] in SUFFIX:
        return float(text[:-1]) * SUFFIX[text[-1]]
    return float(text)


def fmt_num(v, pct=False):
    if v is None or pd.isna(v):
        return '—'
    if pct:
        return f'{v:.2f}%'
    for div, suf in ((1e9, 'B'), (1e6, 'M'), (1e3, 'K')):
        if abs(v) >= div:
            return f'{v / div:.2f}{suf}'
    return f'{v:,.0f}'


def find_file(name):
    for base in ('.', '..'):
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    return None


# ------------------------------------------------------------------------------
# Loading: model + reference dataset
# ------------------------------------------------------------------------------
@st.cache_resource
def load_trained_pipeline():
    path = find_file('best_model_pipeline.joblib')
    return joblib.load(path) if path else None


@st.cache_data
def load_reference():
    """All known influencers with exact values + which split each belongs to.

    The train/validation split is reproduced exactly as in the notebook
    (train_test_split, test_size=0.2, random_state=42), so we can say honestly
    whether the model has seen an influencer during training.
    """
    frames = []
    main_path = find_file('top_insta_influencers_data.csv')
    if main_path:
        main = pd.read_csv(main_path)
        _, val_idx = train_test_split(
            np.arange(len(main)), test_size=0.2, random_state=42
        )
        main['set'] = 'Training'
        main.loc[np.asarray(val_idx), 'set'] = 'Validation'
        frames.append(main)
    ext_path = find_file('indian_influencers_test.csv')
    if ext_path:
        ext = pd.read_csv(ext_path)
        ext['set'] = 'External test'
        frames.append(ext)
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    for c in NUM_COLS:
        df[c] = df[c].map(to_number)
    df['handle'] = (
        df['channel_info'].astype(str).str.strip().str.lower().str.replace('@', '', regex=False)
    )
    df['country_raw'] = df['country']
    df['country'] = df['country'].fillna('Unknown').replace('', 'Unknown')
    df = df.drop_duplicates('handle', keep='first').reset_index(drop=True)
    return df


pipeline = load_trained_pipeline()
ref = load_reference()


def profile_from_ref_row(row):
    return {
        'handle': row['handle'],
        'display_name': f"@{row['handle']}",
        'posts': float(row['posts']),
        'followers': float(row['followers']),
        'avg_likes': float(row['avg_likes']),
        '60_day_eng_rate': float(row['60_day_eng_rate']),
        'new_post_avg_like': float(row['new_post_avg_like']),
        'total_likes': float(row['total_likes']),
        'country': row['country'],
        'country_missing': pd.isna(row['country_raw']) or str(row['country_raw']).strip() == '',
        'actual_score': float(row['influence_score']),
        'rank': int(row['rank']),
        'set': row['set'],
        'source': 'dataset',
    }


def predict_one(profile):
    X = pd.DataFrame([{c: profile[c] for c in FEATURE_COLS}])
    return float(np.clip(pipeline.predict(X)[0], 0, 100))


# ------------------------------------------------------------------------------
# Live Instagram fetch (best effort, only followers / posts / name)
# ------------------------------------------------------------------------------
def _stats_from_posts(followers, posts_list):
    """Average likes / engagement from a list of (likes, comments) of recent posts."""
    likes = [l for l, _ in posts_list if l is not None and l >= 0]
    if not likes or not followers:
        return {}
    comments = [c or 0 for l, c in posts_list if l is not None and l >= 0]
    avg_likes = float(np.mean(likes))
    recent = likes[:6]
    return {
        'avg_likes': avg_likes,
        'new_post_avg_like': float(np.mean(recent)),
        '60_day_eng_rate': float((np.mean(likes) + np.mean(comments)) / followers * 100),
        'n_posts_used': len(likes),
    }


def _fetch_graph_api(handle, token, ig_user_id):
    """Official Instagram Graph API (Business Discovery). Most reliable."""
    fields = (
        f'business_discovery.username({handle})'
        '{followers_count,media_count,name,media.limit(12){like_count,comments_count}}'
    )
    r = requests.get(
        f'https://graph.facebook.com/v19.0/{ig_user_id}',
        params={'fields': fields, 'access_token': token}, timeout=10,
    )
    js = r.json()
    if 'error' in js:
        return None, 'Graph API: ' + js['error'].get('message', 'unknown error')[:160]
    bd = js.get('business_discovery', {})
    followers = float(bd.get('followers_count', 0))
    media = bd.get('media', {}).get('data', [])
    data = {
        'followers': followers,
        'posts': float(bd.get('media_count', 0)),
        'display_name': bd.get('name') or f'@{handle}',
        'method': 'Instagram Graph API',
    }
    data.update(_stats_from_posts(followers, [(m.get('like_count'), m.get('comments_count')) for m in media]))
    return data, None


def _fetch_instaloader(handle):
    """instaloader-based public profile lookup (see instagram_fetch.py)."""
    try:
        return instagram_fetch.fetch_public_profile(handle), None
    except instagram_fetch.ProfileFetchError as exc:
        return None, f'instaloader: {exc}'
    except Exception as exc:  # never let a library error crash the app
        return None, f'instaloader: unexpected {type(exc).__name__}: {exc}'


def _fetch_web_profile(handle):
    """Instagram's public web endpoint used by the website itself (unofficial)."""
    headers = {
        'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                       '(KHTML, like Gecko) Chrome/124.0 Safari/537.36'),
        'X-IG-App-ID': '936619743392459',
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': f'https://www.instagram.com/{handle}/',
        'X-Requested-With': 'XMLHttpRequest',
    }
    r = requests.get(
        'https://www.instagram.com/api/v1/users/web_profile_info/',
        params={'username': handle}, headers=headers, timeout=10,
    )
    if r.status_code != 200:
        return None, (f'web endpoint refused the request (HTTP {r.status_code}) - Instagram often does '
                      'this to scripts, it does not mean the account is missing')
    try:
        user = r.json()['data']['user']
    except (ValueError, KeyError, TypeError):
        return None, 'web endpoint returned no profile data'
    if user is None:
        return None, 'web endpoint returned no user (blocked or unknown handle)'
    if user.get('is_private'):
        return None, 'This account is PRIVATE - stats are not publicly available.'
    followers = float(user['edge_followed_by']['count'])
    media = user.get('edge_owner_to_timeline_media', {})
    edges = media.get('edges', [])
    data = {
        'followers': followers,
        'posts': float(media.get('count', 0)),
        'display_name': user.get('full_name') or f'@{handle}',
        'method': 'Instagram web profile',
    }
    data.update(_stats_from_posts(followers, [
        (e['node'].get('edge_liked_by', {}).get('count'),
         e['node'].get('edge_media_to_comment', {}).get('count')) for e in edges
    ]))
    return data, None


def _fetch_html_meta(handle):
    """Last resort: followers/posts from the page's meta description."""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                             '(KHTML, like Gecko) Chrome/124.0 Safari/537.36',
               'Accept-Language': 'en-US,en;q=0.9'}
    r = requests.get(f'https://www.instagram.com/{handle}/', headers=headers, timeout=10)
    if r.status_code != 200:
        return None, f'profile page refused the request (HTTP {r.status_code})'
    m = re.search(
        r'([\d.,]+[KMBkmb]?)\s+Followers,\s+([\d.,]+[KMBkmb]?)\s+Following,\s+([\d.,]+[KMBkmb]?)\s+Posts',
        r.text)
    if not m:
        return None, 'profile page returned a login wall'
    name_m = re.search(r'og:title"\s+content="([^"]+?)\s*\(@', r.text)
    return {
        'followers': to_number(m.group(1)),
        'posts': to_number(m.group(3)),
        'display_name': html.unescape(name_m.group(1)).strip() if name_m else f'@{handle}',
        'method': 'Instagram profile page (rounded counts)',
    }, None


def clean_handle(text):
    """Accepts 'name', '@name', or a full instagram.com URL and returns the username."""
    t = (text or '').strip().lower()
    m = re.search(r'instagram\.com/([a-z0-9._]+)', t)
    if m:
        t = m.group(1)
    return t.replace('@', '').strip('/ ').split('?')[0]


def fetch_live_instagram(handle):
    """Tries Graph API (if token set) -> instaloader -> web profile JSON -> HTML meta.

    Returns (data_dict | None, error_message | None).
    """
    errors = []
    token = st.session_state.get('ig_token', '').strip() or os.environ.get('IG_ACCESS_TOKEN', '')
    uid = st.session_state.get('ig_uid', '').strip() or os.environ.get('IG_USER_ID', '')
    steps = []
    if token and uid:
        steps.append(lambda: _fetch_graph_api(handle, token, uid))
    steps.append(lambda: _fetch_instaloader(handle))
    steps += [lambda: _fetch_web_profile(handle), lambda: _fetch_html_meta(handle)]
    for step in steps:
        try:
            data, err = step()
        except requests.RequestException as exc:
            data, err = None, f'network error ({type(exc).__name__})'
        except Exception as exc:  # noqa: BLE001
            data, err = None, f'unexpected {type(exc).__name__}: {exc}'
        if data and data.get('followers'):
            return data, None
        if err:
            errors.append(err)
            if 'PRIVATE' in err:
                break
    return None, ' | '.join(errors) or 'unknown error'


def estimated_profile(handle, followers, posts, name, source, live=None):
    """Fills fields Instagram doesn't publish with dataset-median RATIOS.

    These are estimates and are labelled as such in the UI.
    """
    med_like_ratio = float((ref['avg_likes'] / ref['followers']).median())
    med_eng = float(ref['60_day_eng_rate'].median())
    med_new_ratio = float((ref['new_post_avg_like'] / ref['avg_likes']).median())
    avg_likes = followers * med_like_ratio
    live = live or {}
    avg_likes = live.get('avg_likes', avg_likes)
    return {
        'handle': handle,
        'display_name': name,
        'posts': posts,
        'followers': followers,
        'avg_likes': avg_likes,
        '60_day_eng_rate': live.get('60_day_eng_rate', med_eng),
        'new_post_avg_like': live.get('new_post_avg_like', avg_likes * med_new_ratio),
        'total_likes': avg_likes * posts,
        'country': 'Unknown',
        'country_missing': False,
        'actual_score': None,
        'rank': None,
        'set': None,
        'source': source,
    }


# ------------------------------------------------------------------------------
# Per-influencer graphs
# ------------------------------------------------------------------------------
def pct_rank(series, v):
    a = series.dropna().to_numpy()
    return (np.sum(a < v) + 0.5 * np.sum(a == v)) / len(a) * 100


def fig_score_position(profile, pred):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(ref['influence_score'], bins=range(20, 98, 3), color=C_REF, edgecolor='white')
    ax.axvline(pred, color=C_PRED, lw=2.5, label=f'Predicted: {pred:.1f}')
    if profile.get('actual_score') is not None:
        ax.axvline(profile['actual_score'], color=C_ACT, lw=2.5, ls='--',
                   label=f"Actual: {profile['actual_score']:.0f}")
    pr = (ref['influence_score'] < pred).mean() * 100
    ax.set_title(f"Where @{profile['handle']} sits among {len(ref)} influencers "
                 f'(score beats {pr:.0f}%)', fontsize=10, fontweight='bold')
    ax.set_xlabel('Influence score')
    ax.set_ylabel('Number of influencers')
    ax.legend(loc='upper left', fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    return fig


def fig_percentiles(profile):
    labels, vals, texts = [], [], []
    for c in NUM_COLS:
        labels.append(NICE[c])
        vals.append(pct_rank(ref[c], profile[c]))
        texts.append(fmt_num(profile[c], pct=(c == '60_day_eng_rate')))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    colors = [C_PRED if v >= 50 else C_PEER for v in vals]
    bars = ax.barh(labels, vals, color=colors)
    ax.axvline(50, color='#64748B', ls=':', lw=1.2)
    ax.set_xlim(0, 125)
    ax.invert_yaxis()
    for b, v, t in zip(bars, vals, texts):
        tag = f'top {100 - v:.0f}%' if v >= 50 else f'bottom {v:.0f}%'
        ax.text(v + 1.5, b.get_y() + b.get_height() / 2, f'{t}  ({tag})',
                va='center', fontsize=8.5)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel('Percentile vs all known influencers (50 = median)')
    ax.set_title(f"@{profile['handle']}: how each metric ranks", fontsize=10, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    return fig


def fig_scatter(profile, pred):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, col, logx in ((axes[0], 'followers', True), (axes[1], '60_day_eng_rate', False)):
        ax.scatter(ref[col], ref['influence_score'], s=16, color=C_REF, label='Other influencers')
        ax.scatter(profile[col], pred, s=170, marker='*', color=C_PRED,
                   edgecolor='black', zorder=5, label='Predicted')
        if profile.get('actual_score') is not None:
            ax.scatter(profile[col], profile['actual_score'], s=70, marker='D', color=C_ACT,
                       edgecolor='black', zorder=6, label='Actual')
        if logx:
            ax.set_xscale('log')
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_num(v)))
        ax.set_xlabel(NICE[col] + (' (log scale)' if logx else ' (%)'))
        ax.set_ylabel('Influence score')
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(fontsize=8, loc='lower right')
    axes[0].set_title(f"@{profile['handle']}: Followers vs score", fontsize=10, fontweight='bold')
    axes[1].set_title(f"@{profile['handle']}: Engagement vs score", fontsize=10, fontweight='bold')
    fig.tight_layout()
    return fig


def sensitivity(profile, pred):
    """Change in predicted score when each feature is swapped for the dataset median."""
    rows = []
    for c in FEATURE_COLS:
        alt = dict(profile)
        alt[c] = 'Unknown' if c == 'country' else float(ref[c].median())
        rows.append((NICE[c], pred - predict_one(alt)))
    return pd.DataFrame(rows, columns=['Feature', 'Effect']).sort_values('Effect')


def fig_sensitivity(profile, pred):
    sens = sensitivity(profile, pred)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(sens['Feature'], sens['Effect'],
            color=[C_ACT if v >= 0 else '#EF4444' for v in sens['Effect']])
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Score change vs. using the dataset-median value for that feature')
    ax.set_title(f"What drives @{profile['handle']}'s predicted {pred:.1f}?",
                 fontsize=10, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    return fig


def similar_influencers(profile, k=5):
    cols = ['followers', 'avg_likes', '60_day_eng_rate']
    Z = np.log1p(ref[cols])
    mu, sd = Z.mean(), Z.std().replace(0, 1)
    z_ref = (Z - mu) / sd
    z_me = (np.log1p(pd.Series({c: profile[c] for c in cols})) - mu) / sd
    dist = np.sqrt(((z_ref - z_me) ** 2).sum(axis=1))
    cand = ref.assign(_d=dist)
    cand = cand[cand['handle'] != profile['handle']].nsmallest(k, '_d')
    return cand


def fig_similar(profile, pred, peers):
    names = ['@' + h for h in peers['handle']] + [f"@{profile['handle']} (pred.)"]
    scores = list(peers['influence_score']) + [pred]
    colors = [C_PEER] * len(peers) + [C_PRED]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bars = ax.barh(names, scores, color=colors)
    ax.invert_yaxis()
    ax.set_xlim(max(0, min(scores) - 15), 100)
    for b, s in zip(bars, scores):
        ax.text(s + 0.5, b.get_y() + b.get_height() / 2, f'{s:.1f}', va='center', fontsize=9)
    ax.set_xlabel('Influence score')
    ax.set_title('Closest peers by followers, avg likes & engagement', fontsize=10, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    return fig


def show_fig(fig):
    st.pyplot(fig, **STRETCH)
    plt.close(fig)


def render_influencer_charts(profile, pred):
    """All graphs for ONE influencer."""
    st.markdown(f"#### 📈 Graphs for @{profile['handle']}")
    t1, t2, t3, t4, t5 = st.tabs([
        '🎯 Score position', '📊 Metric percentiles', '🔵 Followers & engagement',
        '🔍 Why this score', '👥 Similar influencers',
    ])
    with t1:
        show_fig(fig_score_position(profile, pred))
    with t2:
        show_fig(fig_percentiles(profile))
        st.caption('Blue = above the dataset median, orange = below. Percentiles use all '
                   f'{len(ref)} influencers with known data.')
    with t3:
        show_fig(fig_scatter(profile, pred))
    with t4:
        show_fig(fig_sensitivity(profile, pred))
        st.caption('Each bar shows how much the predicted score changes if that one feature were '
                   'replaced by the dataset median. Features are correlated (e.g. followers and '
                   'total likes), so read this as sensitivity of the model, not as causation.')
    with t5:
        peers = similar_influencers(profile)
        show_fig(fig_similar(profile, pred, peers))
        st.dataframe(
            pd.DataFrame({
                'Handle': peers['handle'].values,
                'Followers': [fmt_num(v) for v in peers['followers']],
                'Avg likes': [fmt_num(v) for v in peers['avg_likes']],
                'Eng. rate': [fmt_num(v, pct=True) for v in peers['60_day_eng_rate']],
                'Actual score': peers['influence_score'].values,
            }),
            **STRETCH, hide_index=True,
        )


# ------------------------------------------------------------------------------
# Result cards shared by pages
# ------------------------------------------------------------------------------
def tier_html(score):
    if score >= 85:
        return ("<span class='badge-tier1'>⭐ Tier 1: Global Mega-Influencer</span>",
                'Massive audience reach with sustained international engagement.')
    if score >= 75:
        return ("<span class='badge-tier2'>🔷 Tier 2: High-Impact Influencer</span>",
                'Strong presence and steady follower interactions.')
    return ("<span class='badge-tier3'>⚡ Tier 3: Emerging / Micro Influencer</span>",
            'Developing audience with active niche engagement.')


def render_prediction_cards(profile, pred):
    c1, c2, c3 = st.columns([1.2, 1.4, 1.4])
    with c1:
        st.markdown(
            f"""<div class='metric-card'>
            <p style='color:#6B7280;margin-bottom:2px;'>Predicted Influence Score</p>
            <h1 style='color:#1E3A8A;font-size:3rem;margin:0;'>{pred:.1f}<span style='font-size:1.3rem;color:#9CA3AF;'>/100</span></h1>
            </div>""", unsafe_allow_html=True)
    with c2:
        badge, desc = tier_html(pred)
        st.markdown(
            f"""<div class='metric-card'><p style='color:#6B7280;margin-bottom:5px;'>Influencer Tier</p>
            {badge}<p style='font-size:0.85rem;color:#4B5563;margin-top:8px;'>{desc}</p></div>""",
            unsafe_allow_html=True)
    with c3:
        act = profile.get('actual_score')
        if act is not None:
            diff = pred - act
            seen = {
                'Training': 'This influencer was in the TRAINING set, so this error is in-sample (optimistic).',
                'Validation': 'This influencer was in the held-out VALIDATION set.',
                'External test': 'This influencer is in the EXTERNAL test set (never seen by the model).',
            }.get(profile.get('set'), '')
            rank_txt = f" (rank #{profile['rank']})" if profile.get('rank') else ''
            st.markdown(
                f"""<div class='metric-card' style='border-left-color:#10B981;'>
                <p style='color:#6B7280;margin-bottom:2px;'>Dataset score{rank_txt}</p>
                <h3 style='margin:0;'>Actual: <b>{act:.0f}</b></h3>
                <p style='margin:3px 0;font-weight:700;color:{"#059669" if abs(diff) < 3 else "#D97706"};'>
                Model error: {diff:+.2f} pts</p>
                <p style='font-size:0.78rem;color:#6B7280;'>{seen}</p></div>""",
                unsafe_allow_html=True)
        else:
            st.markdown(
                """<div class='metric-card' style='border-left-color:#6366F1;'>
                <p style='color:#6B7280;margin-bottom:2px;'>Model Confidence</p>
                <h4 style='margin:0;'>Typical external-test error: ± 1.95 pts</h4>
                <p style='font-size:0.85rem;color:#4B5563;'>No ground-truth score exists for this input.</p></div>""",
                unsafe_allow_html=True)


def render_metric_row(profile):
    m = st.columns(6)
    m[0].metric('Followers', fmt_num(profile['followers']))
    m[1].metric('Posts', fmt_num(profile['posts']))
    m[2].metric('Avg likes', fmt_num(profile['avg_likes']))
    m[3].metric('60-day eng.', f"{profile['60_day_eng_rate']:.2f}%")
    m[4].metric('New-post likes', fmt_num(profile['new_post_avg_like']))
    m[5].metric('Total likes', fmt_num(profile['total_likes']))


# ------------------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------------------
st.sidebar.title('📸 Model Deployment')
page = st.sidebar.radio(
    'Navigation:',
    [
        '🎯 Predict by Instagram ID',
        '📁 Batch / CSV Predictions',
        '📊 Model Evaluation & Diagnostic Plots',
    ],
)
st.sidebar.markdown('---')
st.sidebar.markdown('### **Deployment Details**')
st.sidebar.markdown('**Deployed Best Model:**')
st.sidebar.markdown('🏆 **Gradient Boosting Regressor**')
st.sidebar.markdown('*(80 Trees, Max Depth: 2, Learning Rate: 0.04)*')
st.sidebar.markdown('**Course:** MCA 5141 — ML Mini Project (CO5)')
st.sidebar.markdown('**Team Members:**')
st.sidebar.markdown('• **Vaibhavi Nayak** (`251090700139`)')
st.sidebar.markdown('• **Prakruthi K S** (`2510907000197`)')

with st.sidebar.expander('🔑 Optional: Instagram Graph API (most reliable)'):
    st.caption('Needs an Instagram Business/Creator account linked to a Facebook app. '
               'Works for public Business/Creator accounts and returns real likes data.')
    st.text_input('Access token', type='password', key='ig_token')
    st.text_input('Your IG Business account ID', key='ig_uid')

if pipeline is None or ref.empty:
    st.error('Could not find best_model_pipeline.joblib and/or the dataset CSV files. '
             'Keep app.py in the same folder as these files.')
    st.stop()

# ==============================================================================
# PAGE 1: PREDICT BY INSTAGRAM ID
# ==============================================================================
if page == '🎯 Predict by Instagram ID':
    st.markdown("<h1 class='main-header'>Instagram Influencer Score Predictor</h1>",
                unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Pick any of the <b>205 influencers in our dataset</b> (exact stored data) '
        'or look up <b>any other Instagram ID</b>. Every selection gets its own graphs.</p>',
        unsafe_allow_html=True)

    mode = st.radio(
        'How do you want to choose the influencer?',
        ['📂 From the dataset (exact data)', '🔎 Any other Instagram ID'],
        horizontal=True,
    )

    profile, badge = None, ''

    if mode.startswith('📂'):
        options = ref.sort_values('rank')
        labels = {r['handle']: f"#{int(r['rank'])}  @{r['handle']}" for _, r in options.iterrows()}
        default_idx = list(labels).index('virat.kohli') if 'virat.kohli' in labels else 0
        handle = st.selectbox('Search / select influencer:', list(labels),
                              index=default_idx, format_func=lambda h: labels[h])
        profile = profile_from_ref_row(ref[ref['handle'] == handle].iloc[0])
        badge = "<span class='badge badge-dataset'>📂 Exact data from dataset</span>"

    else:
        c1, c2 = st.columns([3, 1.2])
        with c1:
            typed = st.text_input('Instagram handle (with or without @):', value='mrbeast')
        with c2:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            fetch_btn = st.button('⚡ Fetch live stats', **STRETCH)
        handle = clean_handle(typed)

        if not handle:
            st.info('Type an Instagram handle to begin.')
            st.stop()
        if not re.fullmatch(r'[a-z0-9._]{1,30}', handle):
            st.warning('That does not look like a valid Instagram username '
                       '(letters, numbers, dots and underscores only).')
            st.stop()

        if handle in set(ref['handle']):
            profile = profile_from_ref_row(ref[ref['handle'] == handle].iloc[0])
            badge = "<span class='badge badge-dataset'>📂 Found in dataset — exact data used</span>"
        else:
            live_store = st.session_state.setdefault('live', {})
            if fetch_btn:
                with st.spinner(f'Contacting Instagram for @{handle}...'):
                    data, err = fetch_live_instagram(handle)
                live_store[handle] = {'data': data, 'err': err, 'ts': time.time()}
            result = live_store.get(handle)

            if result and result['data']:
                d = result['data']
                posts_missing = d.get('posts') is None
                base = estimated_profile(
                    handle, d['followers'],
                    float(ref['posts'].median()) if posts_missing else d['posts'],
                    d['display_name'], 'live', live=d)
                if posts_missing:
                    st.warning('Instagram did not return the post count, so Posts is filled with the '
                               'dataset median - please enter the real number below.')
                badge = "<span class='badge badge-live'>🟢 Live from Instagram</span>"
                if 'avg_likes' in d:
                    st.success(f"Live data via **{d['method']}**: followers, posts, and likes/engagement "
                               f"computed from the last {d['n_posts_used']} public posts. "
                               'Total likes is still an estimate (posts × avg likes).')
                else:
                    st.success(f"Live data via **{d['method']}**: followers, posts and name. "
                               'Likes / engagement were not available, so they are estimates below.')
            else:
                if result and result['err']:
                    st.warning(f"Live fetch failed: {result['err']} Enter the numbers manually below.")
                else:
                    st.info('Press **⚡ Fetch live stats** to try reading followers & posts from Instagram, '
                            'or enter the numbers manually below.')
                base = estimated_profile(handle, float(ref['followers'].median()),
                                         float(ref['posts'].median()), f'@{handle}', 'manual')
                badge = "<span class='badge badge-manual'>✍️ Manual / estimated inputs</span>"

            src = base['source']
            st.markdown('##### ✍️ Complete the profile')
            st.caption('Instagram does **not** publicly show avg likes, engagement rate or total likes. '
                       'The fields below are pre-filled with *estimates* (dataset-median ratios) — '
                       'replace them with real numbers from the creator’s insights/analytics tool '
                       'for an accurate prediction.')
            k = f'{handle}_{src}'
            a1, a2, a3 = st.columns(3)
            with a1:
                base['followers'] = st.number_input('Followers', 1000.0, 1e9, float(base['followers']),
                                                    step=1e5, format='%.0f', key=k + 'f')
                base['posts'] = st.number_input('Posts', 1.0, 100000.0, float(base['posts']),
                                                step=10.0, format='%.0f', key=k + 'p')
            with a2:
                base['avg_likes'] = st.number_input('Average likes', 0.0, 5e7, float(base['avg_likes']),
                                                    step=1000.0, format='%.0f', key=k + 'al')
                base['60_day_eng_rate'] = st.number_input('60-day engagement rate (%)', 0.0, 100.0,
                                                          float(base['60_day_eng_rate']), step=0.1,
                                                          format='%.2f', key=k + 'er')
            with a3:
                base['new_post_avg_like'] = st.number_input('New-post avg likes', 0.0, 5e7,
                                                            float(base['new_post_avg_like']),
                                                            step=1000.0, format='%.0f', key=k + 'nl')
                base['total_likes'] = st.number_input('Total likes', 0.0, 1e11, float(base['total_likes']),
                                                      step=1e6, format='%.0f', key=k + 'tl')
            base['country'] = st.selectbox('Country', COUNTRY_CHOICES,
                                           index=COUNTRY_CHOICES.index('Unknown'), key=k + 'c')
            profile = base

    # ---- Profile banner -------------------------------------------------------
    st.markdown('---')
    st.markdown(
        f"""<div class='profile-card'><div style='display:flex;justify-content:space-between;align-items:center;'>
        <div><h3 style='margin:0;color:#1E3A8A;'>{html.escape(profile['display_name'])}
        <span class='insta-tag'>@{html.escape(profile['handle'])}</span></h3>
        <p style='margin:6px 0 0 0;'>{badge}</p></div>
        <div style='text-align:right;'><span style='background:#E2E8F0;padding:4px 10px;border-radius:8px;
        font-size:0.85rem;font-weight:600;color:#111827;'>Market: {html.escape(profile['country'])}</span></div>
        </div></div>""",
        unsafe_allow_html=True)
    if profile.get('country_missing'):
        st.caption('ℹ️ The dataset has no country for this influencer, so — exactly as during training — '
                   'it is treated as “Unknown”.')

    st.markdown('#### 📋 Influencer data used by the model')
    render_metric_row(profile)

    # Optional what-if override for dataset influencers
    if profile['source'] == 'dataset':
        with st.expander('⚙️ What-if: change a metric for this influencer (optional)'):
            use_adj = st.checkbox('Use adjusted values', value=False)
            w1, w2, w3 = st.columns(3)
            adj = dict(profile)
            adj['followers'] = w1.number_input('Followers', 1000.0, 1e9, profile['followers'],
                                               step=1e5, format='%.0f', key=f"w_f_{profile['handle']}")
            adj['avg_likes'] = w2.number_input('Average likes', 0.0, 5e7, profile['avg_likes'],
                                               step=1000.0, format='%.0f', key=f"w_a_{profile['handle']}")
            adj['60_day_eng_rate'] = w3.number_input('60-day eng. rate (%)', 0.0, 100.0,
                                                     profile['60_day_eng_rate'], step=0.1,
                                                     format='%.2f', key=f"w_e_{profile['handle']}")
            if use_adj:
                profile = adj
                profile['actual_score'] = None  # ground truth no longer applies
                st.warning('Hypothetical values in use — the dataset score is hidden because it no '
                           'longer matches these numbers.')

    pred = predict_one(profile)
    st.markdown('---')
    st.markdown(f"### 🎯 Predicted Influence Score for <span class='insta-tag'>@{html.escape(profile['handle'])}</span>",
                unsafe_allow_html=True)
    if profile['source'] != 'dataset':
        st.caption('⚠️ Prediction uses estimated / user-entered engagement fields — accuracy depends on them.')
    render_prediction_cards(profile, pred)
    st.markdown('---')
    render_influencer_charts(profile, pred)

# ==============================================================================
# PAGE 2: BATCH / CSV
# ==============================================================================
elif page == '📁 Batch / CSV Predictions':
    st.markdown("<h1 class='main-header'>Batch Predictions & Verification</h1>", unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Score the external benchmark or your own CSV — and open the '
                'graphs of any single influencer.</p>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(['🇮🇳 5 Indian Influencers Benchmark Test', '📂 Upload Custom CSV File'])

    with tab1:
        st.subheader('External Benchmark: 5 Indian Instagram Influencers')
        st.write('These accounts were kept apart and never shown to the model during training or tuning.')
        ext = ref[ref['set'] == 'External test'].copy()
        ext['pred'] = np.clip(pipeline.predict(ext[FEATURE_COLS]), 0, 100)
        ext['abs_err'] = (ext['pred'] - ext['influence_score']).abs()
        st.dataframe(pd.DataFrame({
            'Instagram Handle': ext['handle'],
            'Followers': ext['followers'].map(fmt_num),
            'Avg Likes': ext['avg_likes'].map(fmt_num),
            '60-Day Eng Rate': ext['60_day_eng_rate'].map(lambda v: fmt_num(v, True)),
            'Actual Score': ext['influence_score'],
            'Predicted Score': ext['pred'].round(2),
            'Absolute Error': ext['abs_err'].round(2),
        }), **STRETCH, hide_index=True)
        st.success(f"**External generalisation MAE:** `{ext['abs_err'].mean():.3f}` score points")

        pick = st.selectbox('Open graphs for:', list(ext['handle']), key='ext_pick')
        prof = profile_from_ref_row(ext[ext['handle'] == pick].iloc[0])
        render_metric_row(prof)
        render_influencer_charts(prof, predict_one(prof))

    with tab2:
        st.subheader('Upload Custom Influencer CSV File')
        st.caption('Required columns: ' + ', '.join(NUM_COLS) + ' (+ optional country, channel_info, influence_score). '
                   'Values like 3.3k, 475.8m, 1.39% are understood.')
        up = st.file_uploader('Upload CSV:', type=['csv'])
        if up is not None:
            raw = pd.read_csv(up)
            st.write('**Preview:**')
            st.dataframe(raw.head(), **STRETCH)
            missing = [c for c in NUM_COLS if c not in raw.columns]
            if missing:
                st.error('Missing required column(s): ' + ', '.join(missing))
            else:
                clean = raw.copy()
                for c in NUM_COLS:
                    clean[c] = clean[c].map(to_number)
                if 'country' not in clean.columns:
                    clean['country'] = 'Unknown'
                clean['country'] = clean['country'].fillna('Unknown')
                clean['pred'] = np.clip(pipeline.predict(clean[FEATURE_COLS]), 0, 100)
                out = raw.copy()
                out['Predicted_Influence_Score'] = clean['pred'].round(2)
                st.success(f'Scored {len(out)} rows.')
                st.dataframe(out, **STRETCH)
                buf = io.StringIO()
                out.to_csv(buf, index=False)
                st.download_button('📥 Download Predictions CSV', buf.getvalue(),
                                   'influencer_score_predictions.csv', 'text/csv')

                names = (clean['channel_info'].astype(str) if 'channel_info' in clean.columns
                         else pd.Series([f'row_{i + 1}' for i in range(len(clean))]))
                idx = st.selectbox('Open graphs for:', range(len(clean)),
                                   format_func=lambda i: f'{i + 1}. {names.iloc[i]}', key='up_pick')
                row = clean.iloc[idx]
                prof = {c: (row[c] if c == 'country' else float(row[c])) for c in FEATURE_COLS}
                prof.update({
                    'handle': str(names.iloc[idx]).strip().lower().replace('@', ''),
                    'actual_score': (float(to_number(row['influence_score']))
                                     if 'influence_score' in clean.columns and pd.notna(row['influence_score'])
                                     else None),
                    'source': 'upload',
                })
                render_metric_row(prof)
                render_influencer_charts(prof, float(row['pred']))

# ==============================================================================
# PAGE 3: MODEL EVALUATION
# ==============================================================================
else:
    st.markdown("<h1 class='main-header'>Model Evaluation & Diagnostic Visualizations</h1>",
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Accuracy, residuals and feature importance of the deployed model.</p>',
                unsafe_allow_html=True)

    plot_dir = 'plots' if os.path.exists('plots') else '../plots'
    figs = [
        ('1. MAE & RMSE', '01_model_mae_rmse_comparison.png',
         'Validation MAE / RMSE of all implemented models.'),
        ('2. 5-Fold CV', '02_cv_mae_error_bars.png', '5-fold cross-validation MAE stability.'),
        ('3. Actual vs Predicted', '03_actual_vs_predicted.png', 'Predictions vs ground truth.'),
        ('4. Residuals', '04_residual_diagnostics.png', 'Residuals vs predicted and their distribution.'),
        ('5. Feature Importance', '05_feature_importance.png', 'Which inputs matter most.'),
        ('6. External Test', '06_external_test_5_influencers.png', 'Predictions on the 5 Indian influencers.'),
    ]
    tabs = st.tabs([f[0] for f in figs] + ['7. Per-influencer check', '8. Metrics table'])
    for tab, (_, fname, cap) in zip(tabs, figs):
        with tab:
            p = os.path.join(plot_dir, fname)
            if os.path.exists(p):
                st.image(p, caption=cap, **STRETCH)
            else:
                st.warning(f'{fname} not found in the plots folder.')

    with tabs[6]:
        st.write('Pick a split and an influencer to see predicted vs actual for that single account.')
        split = st.radio('Split:', ['External test', 'Validation', 'Training'], horizontal=True)
        sub = ref[ref['set'] == split].copy()
        sub['pred'] = np.clip(pipeline.predict(sub[FEATURE_COLS]), 0, 100)
        sub['err'] = sub['pred'] - sub['influence_score']
        st.metric(f'{split} MAE ({len(sub)} influencers)', f"{sub['err'].abs().mean():.2f}")

        who = st.selectbox('Highlight influencer:', list(sub.sort_values('rank')['handle']))
        h = sub[sub['handle'] == who].iloc[0]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(sub['influence_score'], sub['pred'], s=35, color=C_PRED, alpha=0.8)
        lo = min(sub['influence_score'].min(), sub['pred'].min()) - 2
        hi = max(sub['influence_score'].max(), sub['pred'].max()) + 2
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1, label='Perfect prediction')
        ax.scatter(h['influence_score'], h['pred'], s=220, marker='*', color=C_PEER,
                   edgecolor='black', zorder=5, label=f'@{who}')
        ax.set_xlabel('Actual score')
        ax.set_ylabel('Predicted score')
        ax.set_title(f'{split} set: actual vs predicted', fontweight='bold')
        ax.legend()
        ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout()
        cA, cB = st.columns([1.3, 1])
        with cA:
            show_fig(fig)
        with cB:
            st.markdown(f'**@{who}**')
            st.write(f"Actual score: **{h['influence_score']:.0f}**")
            st.write(f"Predicted score: **{h['pred']:.2f}**")
            st.write(f"Error: **{h['err']:+.2f}** points")

    with tabs[7]:
        mpath = find_file('model_evaluation_metrics.csv')
        if mpath:
            st.dataframe(pd.read_csv(mpath).round(3), **STRETCH, hide_index=True)
