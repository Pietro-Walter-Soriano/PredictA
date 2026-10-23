"""Serie A season helpers: a season runs July -> June and is named like FotMob ("2026/2027")."""
from datetime import date

FIRST = 2020   # first season of the archive (2020/2021)


def current_start(today=None):
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


def all_starts(today=None):
    return list(range(FIRST, current_start(today) + 1))


def fotmob(y): return f"{y}/{y + 1}"
