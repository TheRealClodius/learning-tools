# Slack Interface Display Tests

Comprehensive test suite for validating the Slack interface display functionality and ensuring proper information presentation.

## Overview

The `test_slack_interface_display.py` test suite validates:

- **Markdown to Slack Block Kit Conversion**: Ensures proper formatting of markdown content into Slack's visual blocks
- **Streaming Handler Display**: Tests real-time message updates during agent execution
- **Error Display Handling**: Validates proper formatting and presentation of error messages
- **Modal Display Formatting**: Tests execution details modal structure and pagination
- **Mention Resolution**: Validates user and channel mention parsing and display

## Key Features Tested

### 📝 Message Formatting
- Headers, code blocks, lists, and text styling
- Long content splitting for Slack's character limits
- Block Kit structure validation

### 🔄 Real-time Updates
- Streaming message display during agent thinking
- Tool execution progress updates
- Chronological content block management

### ❌ Error Presentation
- Rate limit error formatting
- Authentication error display
- Tool execution error handling
- Parsing error presentation

### 🎛️ Interactive Elements
- Modal execution details formatting
- Pagination for long content
- Button and action element structure

### 🔗 Mention Handling
- User mention resolution (@user)
- Channel mention formatting (#channel)
- Multiple mention scenarios

## Running the Tests

### Quick Run
```bash
python3 run_slack_display_test.py
```

### Direct Test Run
```bash
python3 tests/test_slack_interface_display.py
```

### Test Categories
The test suite runs 22 individual tests across 5 categories:
1. **Markdown Conversion** (6 tests) - Critical for message display
2. **Streaming Handler** (5 tests) - Important for user experience
3. **Error Display** (4 tests) - Important for error handling
4. **Modal Formatting** (3 tests) - Important for detailed views
5. **Mention Resolution** (4 tests) - Nice to have feature

## Test Results

### Success Metrics
- **Success Rate**: Percentage of tests passing
- **Display Quality Score**: Weighted score based on test category importance
- **Performance**: Average execution time per test
- **Block Generation**: Average Slack blocks generated per conversion

### Output Files
Test results are automatically saved to:
```
tests/slack_interface_display_results_[user_id].json
```

Contains:
- Detailed test results for each scenario
- Performance metrics
- Display quality analysis
- Category breakdown

## Test Architecture

### Mock Environment
- Uses `AsyncMock` for Slack API calls
- Simulates real Slack client responses
- No actual API calls made during testing

### Test Categories
Each test category focuses on specific display aspects:

```python
# Example test structure
{
    "test_name": "Simple text formatting",
    "input": "Hello **world**!",
    "expected_blocks": 1,
    "expected_content": "*world*",
    "success": True,
    "execution_time_ms": 0.3
}
```

## Key Validations

### ✅ What Gets Tested
- Slack Block Kit structure compliance
- Markdown to mrkdwn conversion accuracy
- Message update API call patterns
- Error message formatting consistency
- Modal pagination logic
- Mention parsing accuracy

### 📊 Quality Metrics
- **Block Structure**: Proper type and format
- **Content Preservation**: No information loss
- **Performance**: Fast conversion times (<1ms average)
- **Error Handling**: Graceful failure scenarios
- **User Experience**: Proper visual formatting

## Integration with Existing Tests

This test suite complements existing tool and functional tests by focusing specifically on the visual and display aspects of the Slack interface, ensuring users see properly formatted information regardless of the underlying functionality.

The tests use the same patterns as other test files in the project but focus on UI/UX rather than functionality.
