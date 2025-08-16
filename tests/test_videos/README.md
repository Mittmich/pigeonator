# Test Videos Directory

This directory is for user-supplied test videos used in the E2E integration tests.

## Expected Video Files

- `user_supplied_video.mp4` - Main test video with motion and birds
- Additional test videos can be added as needed

## Video Requirements

For best test results, videos should:
- Contain some motion (to trigger motion detection)
- Ideally contain birds or bird-like objects
- Be in a format supported by OpenCV (MP4, AVI, etc.)
- Have reasonable resolution (640x480 or higher recommended)
- Duration of 10-60 seconds for testing purposes

## Usage

Place your test video files in this directory and update the test file paths in `test_e2e_integration.cpp` as needed.

The integration tests will automatically skip if the expected video files are not found.
