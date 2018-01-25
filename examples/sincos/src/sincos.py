"""sincos"""
import math
import time
from kogu import Kogu

# defaults are not really necessary
# but allows to skip their definition in parameters
iterations = 10
step_size = 0.5
amplitude = 2
phase = 0

Kogu.load_parameters()

Kogu.update_parameters({
    "iterations": iterations,
    "step_size": step_size,
    "amplitude": amplitude,
    "phase": phase,
}, output=True)

Kogu.plot(plot_type="line", y_label="sin", series=["sin"], name="Sine")

for i in range(iterations):
    angle = step_size*i
    cos = amplitude*math.cos(angle + phase)
    sin = amplitude*math.sin(angle + phase)
    score = sin + cos
    Kogu.metrics({
        "angle": angle,
        "sin": sin,
        "cos": cos,
    }, iteration=i)
    time.sleep(1)

Kogu.metrics({
    "score": score,
})
