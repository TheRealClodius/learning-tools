# App Home Tab Feature

A comprehensive App Home implementation that provides users with a personalized dashboard experience within the Slack app.

## 🏠 Overview

The App Home tab creates a personalized space for each user where they can:
- View their activity statistics and usage analytics
- Access quick actions for common tasks
- Manage their preferences and settings
- View recent conversation history
- Get help and documentation links
- Report issues or provide feedback

## 🚀 Features

### Personal Dashboard
- **Personalized Greeting**: Time-aware greeting with user's display name
- **Activity Stats**: Total conversations, weekly usage, tools used, and average response time
- **Recent Activity**: Last 5 conversations with preview and channel information

### Quick Actions
- **Start New Chat**: Modal for composing messages to any channel or DM
- **Search Tools**: Overview of available tools and capabilities
- **Settings**: User preferences including timezone and notification settings

### User Settings
- **Timezone Management**: Select from common timezones
- **Notification Preferences**: 
  - Daily summary notifications
  - Tool update notifications
  - Response streaming preferences

### Help & Support
- **Documentation Links**: Quick access to help resources
- **Issue Reporting**: Built-in feedback and bug reporting system

## 🏗️ Implementation Details

### Core Components

#### AppHomeHandler
**Location**: `interfaces/slack/features/app_home/app_home_handler.py`

Main class responsible for:
- Publishing home view layouts
- Handling user interactions
- Managing modal dialogs
- Coordinating with database services

#### Database Integration
**Tables Added**:
- `user_preferences`: Stores user settings and preferences
- `user_activity_stats`: Tracks usage statistics and analytics

**New Methods**:
- `get_user_preferences()` / `save_user_preferences()`
- `get_user_activity_stats()` / `update_user_activity_stats()`

#### Event Handlers
**Core Events**:
- `app_home_opened`: Triggered when user opens the App Home tab
- Action handlers for all interactive elements
- Modal submission handlers for forms

### UI Components

#### Header Section
- Dynamic greeting based on time of day
- Welcome message and overview

#### Quick Actions Bar
- Primary action buttons for common tasks
- Styled with Slack's Block Kit components

#### Statistics Dashboard
- Four-column layout showing key metrics
- Real-time data from database

#### Recent Activity Feed
- Last 5 conversations with previews
- Channel context and timestamps
- "View All" option for extended history

#### Help & Tips
- Documentation links
- Feature tips and usage guidance
- Issue reporting button

### Database Schema

```sql
-- User Preferences
CREATE TABLE user_preferences (
    user_id TEXT PRIMARY KEY,
    notification_prefs TEXT DEFAULT '[]',
    dashboard_layout TEXT DEFAULT 'default',
    theme TEXT DEFAULT 'light',
    app_home_settings TEXT DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User Activity Statistics
CREATE TABLE user_activity_stats (
    user_id TEXT PRIMARY KEY,
    total_messages INTEGER DEFAULT 0,
    messages_this_week INTEGER DEFAULT 0,
    messages_this_month INTEGER DEFAULT 0,
    unique_tools_used INTEGER DEFAULT 0,
    avg_response_time_ms INTEGER DEFAULT 0,
    most_used_tool TEXT DEFAULT '',
    last_activity_date DATE DEFAULT CURRENT_DATE,
    week_start_date DATE DEFAULT CURRENT_DATE,
    month_start_date DATE DEFAULT CURRENT_DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 📋 Configuration

### Required Slack App Permissions

Ensure your Slack app has these scopes:
- `chat:write` - Send messages
- `users:read` - Read user information
- `channels:read` - Access channel information
- `conversations:history` - Read conversation history (for activity tracking)

### Environment Variables

No additional environment variables are required beyond the standard Slack configuration:
- `SLACK_BOT_TOKEN`
- `SLACK_SIGNING_SECRET`

### App Home Tab Setup

1. **Enable App Home**: In your Slack app configuration, navigate to "App Home" and:
   - Enable the Home tab
   - Enable the Messages tab (if desired)
   - Optionally enable "Always Show My Bot as Online"

2. **Event Subscriptions**: Ensure you're subscribed to:
   - `app_home_opened`
   - `message.channels` (for activity tracking)
   - `app_mention` (for agent interactions)

## 🎯 Usage Examples

### Opening App Home
Users can access their App Home by:
1. Clicking the app name in the sidebar
2. Navigating to the "Home" tab in the app
3. The view will automatically load with their personalized dashboard

### Quick Actions
- **Start New Chat**: Opens a modal to compose a message
- **Search Tools**: Shows available capabilities
- **Settings**: Opens preferences management

### Settings Management
Users can customize:
- Timezone for timestamp display
- Notification preferences
- Future: Theme and layout options

## 🔧 Development

### Adding New Dashboard Widgets

1. **Create Widget Method**:
```python
async def _build_new_widget_section(self, user_data: Dict) -> List[Dict]:
    return [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*New Widget*"}
        }
    ]
```

2. **Add to Main View**:
```python
# In _build_home_view method
new_widget_blocks = await self._build_new_widget_section(user_info)
all_blocks = header_blocks + quick_actions_blocks + new_widget_blocks + ...
```

### Adding New Preferences

1. **Update Database Schema**:
Add new columns to `user_preferences` table

2. **Update Settings Modal**:
Add new form elements in `_handle_open_settings`

3. **Handle Submission**:
Update settings submission handler to save new preferences

### Custom Actions

1. **Add Action Handler**:
```python
@self.app.action("new_action")
async def handle_new_action(ack, body, client):
    await self.app_home_handler.handle_app_home_action(ack, body, client, "new_action")
```

2. **Implement Handler**:
```python
async def handle_app_home_action(self, ack, body, client, action_id: str):
    if action_id == "new_action":
        await self._handle_new_action(body, client, user_id)
```

## 🐛 Troubleshooting

### Common Issues

1. **App Home Not Loading**:
   - Check that the Home tab is enabled in app configuration
   - Verify event subscriptions include `app_home_opened`
   - Check logs for publishing errors

2. **Statistics Not Updating**:
   - Ensure `update_user_activity_stats` is called after message processing
   - Check database connectivity
   - Verify table schema is correct

3. **Modal Interactions Failing**:
   - Check that all action handlers are properly registered
   - Verify trigger_id is valid (3-second timeout)
   - Review modal view structure for Block Kit compliance

### Debug Mode

Enable additional logging by setting log level to DEBUG:
```python
logging.getLogger('interfaces.slack.features.app_home').setLevel(logging.DEBUG)
```

## 🚀 Future Enhancements

Planned improvements include:
- **Analytics Dashboard**: More detailed usage analytics
- **Theme Customization**: Light/dark mode options
- **Widget Customization**: User-configurable dashboard layout
- **Conversation Search**: Search through conversation history
- **Export Features**: Download conversation data
- **Team Analytics**: Team-wide usage insights (for admins)

## 📝 API Reference

### AppHomeHandler Methods

#### Core Methods
- `publish_home_view(client, user_id, event=None)`: Publish the main home view
- `handle_app_home_action(ack, body, client, action_id)`: Handle interactive actions

#### Internal Methods
- `_build_home_view(user_info, user_stats, recent_conversations)`: Construct complete home view
- `_build_stats_section(user_stats)`: Create statistics display
- `_build_activity_section(recent_conversations)`: Create activity feed
- `_get_user_stats(user_id)`: Retrieve user statistics
- `_get_recent_conversations(user_id)`: Get conversation history

#### Action Handlers
- `_handle_start_new_chat(body, client, user_id)`: New chat modal
- `_handle_show_tools(body, client, user_id)`: Available tools display
- `_handle_open_settings(body, client, user_id)`: Settings management
- `_handle_view_all_conversations(body, client, user_id)`: Full conversation list
- `_handle_report_issue(body, client, user_id)`: Issue reporting

This App Home implementation provides a solid foundation for user engagement and can be extended with additional features as needed.
