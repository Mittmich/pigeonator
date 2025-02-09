#include "doctest/doctest.h"
#include "orchestration.hpp"
#include <memory>
#include <queue>
#include <set>
#include <thread>
#include "test_utils.hpp"

TEST_CASE("VideoEventManager - Add subscriber") {
    // instantiate mock camera capture
    auto camera_capture = ConstantMockCameraCapture("", 0, 0, 0);
    // instantate mock stream
    auto stream = MockStream(nullptr, &camera_capture);
    VideoEventManager manager(stream);
    auto subscriber = std::make_shared<MockSubscriber>();
    
    manager.add_subscriber(subscriber);
    CHECK(subscriber->event_queue != nullptr);
}