# **Slack Agent Research + Action Capabilities (2025 API)**

## **🎨 Canvas Operations** 
- `canvases.create` - Create collaborative documents/summaries
- `canvases.edit` - Update canvas content with markdown/tables
- `canvases.delete` - Remove canvases
- `canvases.access.set/delete` - Manage canvas permissions
- `canvases.sections.lookup` - Search within canvas sections
- `conversations.canvases.create` - Create channel-specific canvases

## **🔍 Discovery & Search**
### Traditional Search (Rate Limited 2025)
- `search.messages` - Full-text search (20/min)
- `conversations.history` - Channel messages (1/min, 15 msgs for non-Marketplace)
- `conversations.replies` - Thread messages (1/min, 15 msgs for non-Marketplace)

### Real-time Search API (NEW 2025)
- `assistant.search.context` - Live search without data storage
- `assistant.search.info` - Get search capabilities

## **👥 People & Channel Discovery**
- `users.list` - All workspace users
- `users.info` - User profiles, roles, status, timezone
- `users.discoverableContacts.lookup` - Find users by email
- `conversations.list` - All channels/DMs
- `conversations.info` - Channel metadata, topic, purpose
- `conversations.members` - Channel membership
- `team.info` - Workspace information

## **📁 Content Discovery**
- `files.list` - All files in workspace
- `files.info` - File metadata and content
- `pins.list` - Pinned messages (high-value content)
- `bookmarks.list` - Channel bookmarks
- `reactions.list` - Reaction data for sentiment analysis

## **💬 Communication Actions**
- `chat.postMessage` - Send messages
- `chat.postEphemeral` - Send private messages
- `chat.update` - Edit messages
- `chat.delete` - Delete messages
- `chat.scheduleMessage` - Schedule messages
- `chat.meMessage` - Send /me style messages

## **📌 Content Management**
- `pins.add/remove` - Pin/unpin messages
- `bookmarks.add/edit/remove` - Manage channel bookmarks
- `reactions.add/remove` - Add/remove reactions
- `files.upload` - Upload files/documents
- `files.delete` - Delete files

## **🏗️ Channel Management**
- `conversations.create` - Create channels
- `conversations.archive/unarchive` - Archive channels
- `conversations.invite/kick` - Manage channel members
- `conversations.join/leave` - Join/leave channels
- `conversations.rename` - Rename channels
- `conversations.setTopic/setPurpose` - Update channel info

## **🤖 AI Assistant Integration (NEW 2025)**
- `assistant.threads.setStatus` - Update AI thread status
- `assistant.threads.setSuggestedPrompts` - Provide follow-up suggestions
- `assistant.threads.setTitle` - Set thread titles

## **📋 Lists & Data Management**
- `slackLists.items.list` - Get list records
- `slackLists.items.delete/deleteMultiple` - Manage list items

## **📞 Calls & Meetings**
- `calls.add` - Register new calls
- `calls.info` - Get call information
- `calls.update/end` - Manage calls
- `calls.participants.add/remove` - Manage participants

## **⭐ User Engagement**
- `stars.add/remove` - Save/unsave items
- `reminders.add/complete/delete` - Manage reminders
- `dnd.setSnooze/endSnooze` - Manage do not disturb

## **🔧 User Management**
- `users.setPresence` - Set user presence
- `users.profile.set` - Update user profiles
- `usergroups.create/update` - Manage user groups
- `usergroups.users.update` - Update group membership

## **🔐 Enterprise & Admin** (if permissions)
- `admin.conversations.*` - Enterprise channel management
- `admin.users.*` - Enterprise user management
- `admin.teams.*` - Workspace management
- `admin.workflows.*` - Workflow management

## **🔄 Workflows & Automation**
- `workflows.featured.*` - Manage featured workflows
- `functions.*` - Custom function management

## **📊 Metadata & Events**
- `auth.test` - Verify authentication
- `api.test` - Test API connectivity
- `apps.event.authorizations.list` - Get event permissions

## **🌐 Slack Connect**
- `conversations.inviteShared` - Invite to shared channels
- `conversations.acceptSharedInvite` - Accept shared invites
- `conversations.listConnectInvites` - List shared invites

---

**Key 2025 Changes:**
- **Canvas operations** are major new capability
- **Traditional conversation reading** heavily rate-limited (1/min for non-Marketplace)
- **Real-time Search API** provides live discovery alternative
- **AI Assistant APIs** enable enhanced conversational experiences
- **Vector storage + selective live API** strategy now essential