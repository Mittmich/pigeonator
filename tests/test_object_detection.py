from birdhub.object_dection import SingleClassImageSequence

def test_add_detections():
    seq = SingleClassImageSequence(minimum_number_detections=5)
    seq.add_detections(["cat", "dog"], [0.2, 0.3])
    assert seq.has_reached_consensus() == False
    assert seq.get_most_likely_object() == None

def test_has_reached_consensus():
    seq = SingleClassImageSequence()
    seq.add_detections(["cat", "dog", "bird", "elephant", "tiger"], [0.1, 0.2, 0.3, 0.4, 0.5])
    assert seq.has_reached_consensus() == True

def test_get_most_likely_object():
    seq = SingleClassImageSequence()
    seq.add_detections(["cat", "dog"], [0.1, 0.2])
    assert seq.get_most_likely_object() == None
    seq.add_detections(["bird", "elephant", "tiger"], [0.3, 0.4, 0.5])
    assert seq.get_most_likely_object() == "tiger"

def test_multiple_detections():
    seq = SingleClassImageSequence(minimum_number_detections=2)
    seq.add_detections(["pigeon", "pigeon", "crow"], [0.2, 0.2, 0.5])
    assert seq.get_most_likely_object() == "crow"
    seq.add_detections(["pigeon", "pigeon"], [0.5, 0.5])
    assert seq.get_most_likely_object() == "pigeon"
