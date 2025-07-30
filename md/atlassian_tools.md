# **Atlassian Agent Tool Capabilities**

Based on your codebase and the Atlassian API, here are all the available tools:

## **🎫 Jira Operations**

### **Issue Management**
- `jira_search` - Search Jira issues using JQL queries
- `get_jira_issues` - Retrieve specific issues by key/ID
- `create_jira_issue` - Create new tickets/issues
- `update_jira_issue` - Edit existing issues
- `delete_jira_issue` - Delete issues
- `assign_jira_issue` - Assign issues to users
- `transition_jira_issue` - Move issues through workflow states

### **Issue Details & Metadata**
- `get_issue_comments` - Retrieve all comments on an issue
- `add_issue_comment` - Add comments to issues
- `get_issue_attachments` - Get file attachments
- `add_issue_attachment` - Upload files to issues
- `get_issue_watchers` - Get users watching an issue
- `get_issue_links` - Get linked/related issues
- `link_jira_issues` - Create relationships between issues

### **Project & Structure**
- `get_jira_projects` - List all accessible projects
- `get_project_versions` - Get project versions/releases
- `get_project_components` - Get project components
- `get_issue_types` - Get available issue types
- `get_priorities` - Get priority levels
- `get_statuses` - Get workflow statuses

### **Advanced Jira**
- `get_jira_filters` - Get saved search filters
- `run_jira_filter` - Execute saved filters
- `get_dashboards` - Get Jira dashboards
- `get_sprint_info` - Get Agile sprint details (if Agile enabled)
- `get_board_info` - Get Kanban/Scrum board details

## **📄 Confluence Operations**

### **Page Management**
- `confluence_search` - Search Confluence pages and content
- `get_confluence_pages` - Retrieve specific pages
- `create_confluence_page` - Create new pages
- `update_confluence_page` - Edit existing pages
- `delete_confluence_page` - Delete pages
- `get_page_content` - Get page content in various formats

### **Space & Structure**
- `get_confluence_spaces` - List all accessible spaces
- `get_space_content` - Get all content in a space
- `get_page_children` - Get child pages
- `get_page_ancestors` - Get parent page hierarchy
- `get_page_labels` - Get page tags/labels

### **Content & Collaboration**
- `get_page_comments` - Get page comments
- `add_page_comment` - Add comments to pages
- `get_page_attachments` - Get file attachments
- `add_page_attachment` - Upload files to pages
- `get_page_history` - Get page revision history
- `get_page_restrictions` - Get page permissions

## **👥 User & Permission Management**

### **User Operations**
- `get_user_info` - Get user profile details
- `search_users` - Find users by name/email
- `get_user_groups` - Get user group memberships
- `get_user_permissions` - Get user permission levels

### **Group Management**
- `get_groups` - List all groups
- `get_group_members` - Get users in a group
- `add_user_to_group` - Add users to groups
- `remove_user_from_group` - Remove users from groups

## **📊 Analytics & Reporting**

### **Project Analytics**
- `get_project_statistics` - Project metrics and stats
- `get_issue_statistics` - Issue distribution and trends
- `get_user_activity` - User contribution metrics
- `get_time_tracking` - Time logging data

### **Advanced Reporting**
- `run_custom_jql` - Execute complex JQL queries
- `get_version_report` - Version/release reports
- `get_component_report` - Component usage reports
- `get_worklog_report` - Time tracking reports

## **🔧 Administrative Tools**

### **System Information**
- `get_atlassian_status` - Service health and status
- `get_server_info` - Instance information
- `get_configuration` - System configuration details
- `get_application_properties` - App settings

### **Workflow & Customization**
- `get_workflows` - Get workflow schemes
- `get_custom_fields` - Get custom field definitions
- `get_field_configurations` - Get field config schemes
- `get_notification_schemes` - Get notification settings

## **🔗 Integration & Automation**

### **Webhooks & Events**
- `create_webhook` - Set up event notifications
- `get_webhooks` - List configured webhooks
- `delete_webhook` - Remove webhooks

### **REST API Extensions**
- `execute_automation_rule` - Trigger Jira automation
- `get_app_properties` - Get app-specific settings
- `bulk_operations` - Batch processing operations

## **🎯 Your Current Implementation Focus**

Based on your codebase, you currently have these **core functions** implemented:

### **Primary Tools**:
- `jira_search` - JQL-based issue search
- `get_jira_issues` - Retrieve specific issues
- `confluence_search` - Page and content search  
- `get_confluence_pages` - Page retrieval
- `get_jira_projects` - Project listing
- `get_confluence_spaces` - Space listing
- `get_atlassian_status` - System health check

### **Enhanced Discovery Tools**:
- `issue_details` - Deep issue analysis
- `page_content` - Full page content retrieval
- `user_analysis` - User activity and contribution analysis

## **🚀 Most Valuable for Your Agent**

### **High-Impact Research Tools**:
1. **`jira_search`** - Complex JQL queries for pattern discovery
2. **`confluence_search`** - Document and knowledge base search
3. **`get_issue_comments`** - Conversation and decision tracking
4. **`get_page_history`** - Document evolution analysis
5. **`get_user_activity`** - Expertise and contribution mapping

### **Action-Oriented Tools**:
1. **`create_jira_issue`** - Task and follow-up creation
2. **`add_issue_comment`** - Progress updates and communication
3. **`create_confluence_page`** - Documentation and summary creation
4. **`link_jira_issues`** - Relationship mapping

The Atlassian ecosystem gives your agent comprehensive **project intelligence** capabilities - from tactical issue tracking to strategic documentation analysis!