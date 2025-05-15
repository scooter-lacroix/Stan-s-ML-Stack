# CI/CD Pipeline Setup for Stan's ML Stack

This document explains how to set up the CI/CD pipeline for automatically updating the Docker image with security patches.

## Required GitHub Secrets

The workflow requires the following secrets to be set up in your GitHub repository:

### 1. Docker Hub Credentials

- `DOCKERHUB_USERNAME`: Your Docker Hub username
- `DOCKERHUB_TOKEN`: A Docker Hub access token (not your password)

To create a Docker Hub access token:
1. Log in to [Docker Hub](https://hub.docker.com/)
2. Go to Account Settings → Security
3. Click "New Access Token"
4. Give it a name like "GitHub Actions"
5. Copy the token immediately (it won't be shown again)

### 2. Optional: Slack Notifications

- `SLACK_WEBHOOK_URL`: A Slack webhook URL for notifications

To create a Slack webhook:
1. Go to [Slack API Apps](https://api.slack.com/apps)
2. Create a new app or use an existing one
3. Enable "Incoming Webhooks"
4. Create a new webhook URL for your workspace
5. Copy the webhook URL

## Setting Up Secrets in GitHub

1. Go to your GitHub repository
2. Click on "Settings" → "Secrets and variables" → "Actions"
3. Click "New repository secret"
4. Add each of the secrets mentioned above

## Workflow Details

The workflow:
- Runs automatically every Monday at 2:00 AM UTC
- Can be triggered manually from the Actions tab
- Builds and pushes the Docker image with security updates
- Creates a GitHub issue with a summary of the changes
- Uploads detailed security reports as artifacts
- Sends a Slack notification (if configured)

## Manual Triggering

To manually trigger the workflow:
1. Go to the "Actions" tab in your GitHub repository
2. Select "Weekly Docker Security Update" from the workflows list
3. Click "Run workflow"
4. Optionally provide a reason for the manual trigger
5. Click "Run workflow" again

## Customizing the Schedule

To change the schedule, edit the cron expression in `.github/workflows/docker-security-update.yml`:

```yaml
on:
  schedule:
    # Format: minute hour day-of-month month day-of-week
    - cron: '0 2 * * 1'  # Every Monday at 2:00 AM UTC
```

Common examples:
- Daily at midnight UTC: `0 0 * * *`
- Every Sunday at 3:00 AM UTC: `0 3 * * 0`
- First day of each month: `0 0 1 * *`
- Every 6 hours: `0 */6 * * *`
