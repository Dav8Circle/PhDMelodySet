from mtypes import MelodyTokenizer, FantasticTokenizer

def test_get_notes():
    # Initialize tokenizer
    tokenizer = MelodyTokenizer()
    
    # Test case with a simple melody of 3 notes
    pitches = [60, 62, 64]  # C4, D4, E4
    starts = [0.0, 1.0, 2.0]
    ends = [0.8, 1.8, 2.8]

    notes = tokenizer.get_notes(pitches, starts, ends)

    # Test we get the right number of notes
    assert len(notes) == 3

    # Test first note
    assert notes[0].pitch == 60
    assert notes[0].start == 0.0
    assert notes[0].end == 0.8
    assert notes[0].duration == 0.8
    assert notes[0].ioi == 1.0  # Time to next note
    assert notes[0].ioi_ratio is None  # First note has no IOI ratio

    # Test middle note
    assert notes[1].pitch == 62
    assert notes[1].start == 1.0
    assert notes[1].end == 1.8
    assert notes[1].duration == 0.8
    assert notes[1].ioi == 1.0
    assert notes[1].ioi_ratio == 1.0  # Equal IOIs

    # Test last note
    assert notes[2].pitch == 64
    assert notes[2].start == 2.0
    assert notes[2].end == 2.8
    assert notes[2].duration == 0.8
    assert notes[2].ioi is None  # Last note has no next IOI
    assert notes[2].ioi_ratio is None  # Last note has no IOI ratio

def test_segment_melody():
    # Initialize tokenizer with default gap of 1.0
    tokenizer = FantasticTokenizer()
    
    # Test case with two phrases separated by a gap > 1.0
    pitches = [60, 62, 64, 65, 67]  # C4, D4, E4, F4, G4
    starts = [0.0, 0.5, 1.0, 3.0, 3.5]  # Gap of 1.6 between end of E4 and start of F4
    ends =   [0.4, 0.9, 1.4, 3.4, 3.9]
    
    # Create Note objects first
    notes = MelodyTokenizer().get_notes(pitches, starts, ends)
    phrases = tokenizer.segment_melody(notes)
    
    # Should be split into two phrases
    assert len(phrases) == 2
    
    # First phrase should have 3 notes
    assert len(phrases[0]) == 3
    assert phrases[0][0].pitch == 60
    assert phrases[0][1].pitch == 62
    assert phrases[0][2].pitch == 64
    
    # Second phrase should have 2 notes
    assert len(phrases[1]) == 2
    assert phrases[1][0].pitch == 65
    assert phrases[1][1].pitch == 67
    
    # Test with a smaller gap that shouldn't split the phrase
    starts = [0.0, 0.5, 1.0, 1.8, 2.3]  # All gaps < 1.0
    ends =   [0.4, 0.9, 1.4, 2.2, 2.7]
    
    # Create Note objects first
    notes = MelodyTokenizer().get_notes(pitches, starts, ends)
    phrases = tokenizer.segment_melody(notes)
    
    # Should be a single phrase
    assert len(phrases) == 1
    assert len(phrases[0]) == 5

def test_tokenize_phrase():
    tokenizer = FantasticTokenizer()

    # Test case with various intervals and IOI ratios
    pitches = [60, 62, 65, 64, 60]  # C4, D4, F4, E4, C4
    starts = [0.0, 1.0, 1.5, 2.0, 4.0]  # Different IOI ratios
    ends =   [0.8, 1.3, 1.8, 2.8, 4.8]

    notes = MelodyTokenizer().get_notes(pitches, starts, ends)
    tokens = tokenizer.tokenize_phrase(notes)

    # Should have len(notes)-1 tokens
    assert len(tokens) == 4

    # Check each token (interval class, ioi ratio class)
    # Token 1: +2 semitones (major second up), IOI ratio None
    assert tokens[0] == ('u2', "q")  # C4 to D4, 1.0 to 0.5 beat IOI

    # Token 2: +3 semitones (minor third up), IOI ratio 0.5
    assert tokens[1] == ('u3', 'e')  # D4 to F4, 0.5 to 0.5 beat IOI

    # Token 3: -1 semitone (minor second down), IOI ratio 1.0 (equal)
    assert tokens[2] == ('d2', 'l')  # F4 to E4, 0.5 to 2.0 beat IOI

    # Token 4: -4 semitones (major third down), IOI ratio 4.0 (longer)
    assert tokens[3] == ('d3', None)  # E4 to C4, long gap
