# PICTION EXHICTURES AT AN EXHIBIT AN EXHIBITION
# Copyright (C) 2013  Andreas Jansson

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import collections
import numpy as np
from scikits.learn.linear_model import LogisticRegression
import scipy.signal
import matplotlib.pyplot as plt
import midi
import time
import sys

def get_notes():
    # downloaded from http://www.piano-midi.de/muss.htm
    filename = 'muss_1.mid'
    m = midi.read_midifile(filename)
    m.make_ticks_abs()
    tick = 120.0
    notes = np.array([(n.pitch, int(round(n.tick / tick)))
                      for n in m[1]
                      if type(n) == midi.events.NoteOnEvent
                      and n.velocity > 0
                      and n.pitch > 0])

    note_map = collections.defaultdict(list)
    max_pitch = 0
    min_pitch = 127
    for pitch, t in notes:
        note_map[t].append(pitch)
        if pitch > max_pitch:
            max_pitch = pitch
        elif pitch < min_pitch:
            min_pitch = pitch

    max_time = max(note_map.keys())

    output_units = max_pitch - min_pitch + 1
    output = np.zeros((max_time, output_units))
    for t in range(max_time):
        if t in note_map:
            for i in note_map[t]:
                output[t, i - min_pitch] = 1

    return output, min_pitch

def activation(state, on):
    if on:
        return 1
    else:
        return state / 1.4

def repeat_classes(train_data, target, neg_rep, pos_rep):
    negative = train_data[target == 0]
    positive = train_data[target == 1]

    if len(negative) > len(positive) and len(positive) > 0:
        positive = np.tile(positive, (int(len(negative) / (len(positive) * 2)), 1))

    train_data = np.vstack((np.tile(negative, (neg_rep, 1)), np.tile(positive, (pos_rep, 1))))
    target = [0] * (len(negative) * neg_rep) + [1] * (len(positive) * pos_rep)

    return train_data, target

def main():
    notes, min_pitch = get_notes()
    duration, n_pitches = notes.shape
    data = np.zeros((n_pitches, duration, n_pitches))
    classes = np.zeros((n_pitches, duration))
    state = np.zeros((n_pitches))

    for t, pitches in enumerate(notes):
        for i, on in enumerate(pitches):
            data[i, t] = state
            classes[i, t] = on
        
        for i, on in enumerate(pitches):
            state[i] = activation(state[i], on)

    models = []
    for i in xrange(n_pitches):
        model = LogisticRegression('l2', tol=0.1)
        train_data, target = repeat_classes(data[i, :], classes[i, :], 10, 10)
        model.fit(train_data, target)
        models.append(model)

    duration *= 3
    predicted = np.zeros((n_pitches, duration))
    state = np.zeros((n_pitches))
    state[21] = 1
    for t in xrange(1, duration):
        sys.stdout.write('%d\r' % t)
        sys.stdout.flush()
        current_state = state.reshape((1, state.shape[0]))
        for i in xrange(n_pitches):
            on = models[i].predict(current_state)[0]
            predicted[i, t] = on
            state[i] = activation(state[i], on)

        state += (np.random.random((n_pitches)) - .5) * .1

    write(predicted, min_pitch)

def write(predicted, min_pitch):
    from midiutil.MidiFile import MIDIFile
    m = MIDIFile(1)
    m.addTempo(0, 0, 70)
    for t, pitches in enumerate(predicted.T):
        for i, on in enumerate(pitches):
            note = i + min_pitch
            if on:
                m.addNote(0, 0, note, t / 8.0, 1 / 8.0, 100)
    with open('out.mid', 'wb') as f:
        m.writeFile(f)

    # generated with timidity --reverb=d --default-program 4 out.mid
    # fluidsynth

def play(predicted):
    import alsaseq, alsamidi

    alsaseq.client('andreas', 1, 1, True)
    alsaseq.connectto(1, 20, 0)
    alsaseq.start()

    for pitches in predicted.T:
        for i, on in enumerate(pitches):
            note = i + 50
            alsaseq.output(alsamidi.noteoffevent(0, note, 100))
            if on:
                alsaseq.output(alsamidi.noteonevent(0, note, 100))
        time.sleep(.1)

if __name__ == '__main__':
    main()
