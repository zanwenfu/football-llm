#!/usr/bin/env python3
"""Verify API configuration and check quota."""

from dotenv import load_dotenv
load_dotenv()
import os

print('='*60)
print('CONFIGURATION CHECK')
print('='*60)

# Check environment variables
rate_limit = os.getenv('REQUESTS_PER_KEY_PER_MINUTE', '10')
daily_limit = os.getenv('DAILY_REQUEST_LIMIT', '100')
api_key = os.getenv('API_FOOTBALL_KEY', '')

print(f'Rate limit: {rate_limit} requests/minute')
print(f'Daily limit: {daily_limit} requests/day')
print(f'API Key: {api_key[:8]}...{api_key[-4:]}')

# Check API connection and quota
print()
print('='*60)
print('API STATUS CHECK')
print('='*60)

from api_client import api_client
status = api_client.get_account_status()
response = status.get('response', {})

account = response.get('account', {})
subscription = response.get('subscription', {})
requests_info = response.get('requests', {})

print(f'Account: {account.get("firstname", "N/A")} {account.get("lastname", "N/A")}')
print(f'Email: {account.get("email", "N/A")}')
print(f'Plan: {subscription.get("plan", "N/A")}')
print(f'Plan end: {subscription.get("end", "N/A")}')
print()
print(f'Requests today: {requests_info.get("current", "N/A")}')
print(f'Daily limit: {requests_info.get("limit_day", "N/A")}')

current = requests_info.get('current', 0)
limit = requests_info.get('limit_day', 0)
remaining = limit - current
print(f'Remaining: {remaining}')

# Calculate capacity
print()
print('='*60)
print('CAPACITY ESTIMATE')
print('='*60)

# Estimate: ~35 players per team, 3 seasons = 105 API calls per team + 1 for squad
calls_per_team = 35 * 3 + 1  # ~106 calls per team
teams_can_scrape = remaining // calls_per_team

print(f'Calls per team: ~{calls_per_team}')
print(f'Teams we can scrape: ~{teams_can_scrape}')
print(f'At 300 req/min, time per team: ~{calls_per_team / 300 * 60:.1f} seconds')

if remaining > 3000:
    print('\n✅ READY TO SCRAPE!')
else:
    print('\n⚠️ Consider waiting for quota reset before full scrape')
