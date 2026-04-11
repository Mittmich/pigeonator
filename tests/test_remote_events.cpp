#include "doctest/doctest.h"
#include "remote_events.hpp"
#include "events.hpp"
#include "timestamp_utils.hpp"
#include <memory>
#include <queue>

// Helper: build a DetectionEvent with metadata
static std::shared_ptr<DetectionEvent> make_detection_with_meta(
    const std::string& label, float confidence)
{
    auto ts = test_now();
    auto frame_event  = std::make_shared<FrameEvent>(ts, std::nullopt);
    std::map<std::string, std::string> meta{
        {"most_likely_object", label},
        {"mean_confidence",    std::to_string(confidence)},
    };
    Detection det(ts, frame_event,
        std::vector<std::string>{label},
        std::vector<float>{confidence},
        std::nullopt, std::nullopt, meta);
    return std::make_shared<DetectionEvent>(ts, std::vector<Detection>{det});
}

// Helper: build an EffectorActionEvent
static std::shared_ptr<EffectorActionEvent> make_effector_event(const std::string& action) {
    std::map<std::string, std::string> meta{{"triggered_class", "Pigeon"}};
    return std::make_shared<EffectorActionEvent>(test_now(), action, meta);
}

TEST_CASE("EventDispatcher: listens to DETECTION, EFFECTOR_ACTION, and RECORDING_STOPPED") {
    EventDispatcher dispatcher("http://localhost:1");
    auto listening = dispatcher.listening_to();
    CHECK(listening.count(EventType::DETECTION)        == 1);
    CHECK(listening.count(EventType::EFFECTOR_ACTION)  == 1);
    CHECK(listening.count(EventType::RECORDING_STOPPED) == 1);
    CHECK(listening.size() == 3);
}

TEST_CASE("EventDispatcher: start and stop without requests") {
    EventDispatcher dispatcher("http://localhost:1");
    dispatcher.start();
    dispatcher.stop();
    // No assertion — just must not deadlock or crash
}

TEST_CASE("EventDispatcher: notify enqueues events without blocking") {
    // unreachable server - HTTP errors are logged but must not throw or block
    EventDispatcher dispatcher("http://localhost:19999");
    dispatcher.start();

    auto det_event = make_detection_with_meta("Pigeon", 0.9f);
    auto eff_event = make_effector_event("effect_activated");

    // notify() must return immediately (event is queued, not sent inline)
    dispatcher.notify(det_event);
    dispatcher.notify(eff_event);

    dispatcher.stop();
}

TEST_CASE("EventDispatcher: ignores NEW_FRAME events") {
    EventDispatcher dispatcher("http://localhost:19999");
    dispatcher.start();

    auto ts    = test_now();
    auto frame = std::make_shared<FrameEvent>(ts, std::nullopt);
    // NEW_FRAME is not in listening_to(), so VideoEventManager won't call notify().
    // Verify listening_to() does not include NEW_FRAME.
    CHECK(dispatcher.listening_to().count(EventType::NEW_FRAME) == 0);

    dispatcher.stop();
}

TEST_CASE("EventDispatcher: set_event_queue does not throw") {
    EventDispatcher dispatcher("http://localhost:1");
    auto q = std::make_shared<std::queue<std::shared_ptr<Event>>>();
    dispatcher.set_event_queue(q);
    // Nothing to assert — interface satisfied
}
