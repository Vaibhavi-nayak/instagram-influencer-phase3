"""
instagram_fetch.py
-------------------
Fetches PUBLIC Instagram profile stats for a username using the unofficial
`instaloader` library (no login, no API key) and converts them into the
features used by the Phase 3 influence-score model.

HONEST LIMITATIONS
1. Not Instagram's official API. It reads the same public data a browser sees.
2. Instagram rate-limits / blocks anonymous requests (401/429), especially after
   several lookups in a row. Wait a few minutes or use manual entry.
3. Public accounts only. Private accounts fail with a clear message.
4. May go against Instagram's Terms of Service - personal / educational use only.
5. Instagram does not publish "total likes" or the dataset's exact 60-day
   engagement definition; those are derived/estimated in app.py and labelled.
"""

import numpy as np

try:
    import instaloader
except ImportError:  # app still works without it (falls back to other methods)
    instaloader = None

MAX_POSTS = 12  # keep it fast and reduce the chance of rate limiting


class ProfileFetchError(Exception):
    """Raised whenever a profile can't be fetched, with a human-readable reason."""


def fetch_public_profile(username: str) -> dict:
    """Returns followers, posts, display_name, avg_likes, new_post_avg_like,
    60_day_eng_rate and n_posts_used for a PUBLIC account."""
    if instaloader is None:
        raise ProfileFetchError('instaloader is not installed (pip install instaloader).')
    username = username.strip().lstrip('@')
    if not username:
        raise ProfileFetchError('Please enter a username.')

    L = instaloader.Instaloader(
        download_pictures=False, download_videos=False, download_video_thumbnails=False,
        download_geotags=False, download_comments=False, save_metadata=False,
        compress_json=False, quiet=True,
    )
    try:
        profile = instaloader.Profile.from_username(L.context, username)
    except instaloader.exceptions.ProfileNotExistsException:
        raise ProfileFetchError(
            f"instaloader found no account '{username}' (this can also happen when "
            'Instagram blocks the request).')
    except instaloader.exceptions.ConnectionException as e:
        raise ProfileFetchError(f'Instagram blocked or rate-limited the request ({e}).')
    except Exception as e:  # noqa: BLE001
        raise ProfileFetchError(f'Could not fetch this profile: {e}')

    if getattr(profile, 'is_private', False):
        raise ProfileFetchError(f'@{username} is a PRIVATE account - stats are not publicly available.')

    # Anonymous requests sometimes return PARTIAL profile data (fields set to None).
    followers_raw = getattr(profile, 'followers', None)
    if followers_raw is None:
        raise ProfileFetchError(
            'Instagram returned incomplete profile data (followers missing) - this is a '
            'partial block of anonymous requests. Wait a few minutes and retry.')
    followers = float(followers_raw)
    data = {
        'followers': followers,
        'display_name': getattr(profile, 'full_name', None) or f'@{username}',
        'method': 'instaloader (public profile)',
    }
    media_count = getattr(profile, 'mediacount', None)
    if media_count is not None:
        data['posts'] = float(media_count)

    likes, comments = [], []
    try:
        for i, post in enumerate(profile.get_posts()):
            if i >= MAX_POSTS:
                break
            if post.likes and post.likes > 0:  # hidden likes come back as 0/-1
                likes.append(post.likes)
                comments.append(post.comments or 0)
    except Exception:  # noqa: BLE001
        # Profile stats are still valid; engagement just stays estimated.
        return data

    if likes and followers:
        data.update({
            'avg_likes': float(np.mean(likes)),
            'new_post_avg_like': float(np.mean(likes[:6])),
            '60_day_eng_rate': float((np.mean(likes) + np.mean(comments)) / followers * 100),
            'n_posts_used': len(likes),
        })
    return data
