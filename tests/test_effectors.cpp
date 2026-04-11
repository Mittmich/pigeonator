#include "doctest/doctest.h"
#include "events.hpp"
#include "mock_effector.hpp"
#include "sound_effector.hpp"
#include "timestamp_utils.hpp"
#include <memory>
#include <queue>
#include <thread>
#include <chrono>

// Helper: build a DetectionEvent with a given label
static std::shared_ptr<DetectionEvent> make_detection_event(const std::string& label) {
    auto ts = test_now();
    auto frame_event = std::make_shared<FrameEvent>(ts, std::nullopt);
    Detection det(ts, frame_event,
        std::vector<std::string>{label},
        std::vector<float>{0.9f},
        std::nullopt, std::nullopt, std::nullopt);
    return std::make_shared<DetectionEvent>(ts, std::vector<Detection>{det});
}

// Helper: attach an event queue to a subscriber
static std::shared_ptr<std::queue<std::shared_ptr<Event>>> attach_queue(Subscriber& s) {
    auto q = std::make_shared<std::queue<std::shared_ptr<Event>>>();
    s.set_event_queue(q);
    return q;
}

TEST_CASE("MockEffector: listens to DETECTION events only") {
    MockEffector effector({"Pigeon"});
    auto listening = effector.listening_to();
    CHECK(listening.count(EventType::DETECTION) == 1);
    CHECK(listening.size() == 1);
}

TEST_CASE("MockEffector: triggers on matching target class") {
    MockEffector effector({"Pigeon"});
    auto q = attach_queue(effector);

    effector.notify(make_detection_event("Pigeon"));

    REQUIRE(!q->empty());
    auto event = q->front();
    CHECK(event->type == EventType::EFFECTOR_ACTION);
}

TEST_CASE("MockEffector: does not trigger on non-matching class") {
    MockEffector effector({"Pigeon"});
    auto q = attach_queue(effector);

    effector.notify(make_detection_event("Sparrow"));

    CHECK(q->empty());
}

TEST_CASE("MockEffector: empty target_classes triggers for any class") {
    MockEffector effector({});  // no filter = all classes
    auto q = attach_queue(effector);

    effector.notify(make_detection_event("Anything"));

    CHECK(!q->empty());
}

TEST_CASE("MockEffector: cooldown prevents rapid re-trigger") {
    MockEffector effector({"Pigeon"}, std::chrono::seconds(60));
    auto q = attach_queue(effector);

    effector.notify(make_detection_event("Pigeon"));
    CHECK(q->size() == 1);

    effector.notify(make_detection_event("Pigeon"));
    CHECK(q->size() == 1);  // Second call blocked by cooldown
}

TEST_CASE("MockEffector: triggers again after cooldown expires") {
    MockEffector effector({"Pigeon"}, std::chrono::seconds(0));
    auto q = attach_queue(effector);

    effector.notify(make_detection_event("Pigeon"));
    effector.notify(make_detection_event("Pigeon"));

    CHECK(q->size() == 2);
}

TEST_CASE("EffectorActionEvent: carries action string and metadata") {
    auto ts = test_now();
    std::map<std::string, std::string> meta{{"triggered_class", "Pigeon"}};
    EffectorActionEvent ev(ts, "effect_activated", meta);

    CHECK(ev.type == EventType::EFFECTOR_ACTION);
    CHECK(ev.get_action() == "effect_activated");
    CHECK(ev.get_meta_data().at("triggered_class") == "Pigeon");
}

TEST_CASE("SoundEffector: triggers on matching class (non-blocking)") {
    // Use a non-existent sound file - we just verify the event is emitted
    SoundEffector effector({"Pigeon"}, "/nonexistent/sound.mp3");
    auto q = attach_queue(effector);

    effector.notify(make_detection_event("Pigeon"));

    // Give the detached thread a moment, but the event emission is synchronous
    CHECK(!q->empty());
    CHECK(q->front()->type == EventType::EFFECTOR_ACTION);
}
