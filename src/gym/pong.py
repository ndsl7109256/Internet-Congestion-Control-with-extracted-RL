# Copyright 2017-2018 MIT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import gym


def get_pong_symbolic(obs):
    '''
    # preprocessing
    obs = obs[:83,:,:]
    obs = [obs[:,:,3], obs[:,:,2], obs[:,:,1], obs[:,:,0]]
    b = [_get_ball(ob)[0] for ob in obs]
    c = [_get_ball(ob)[1] for ob in obs]
    p = [_get_our_paddle(ob) for ob in obs]

    # new state
    symb = []

    # position
    symb.append(b[0] - p[0])
    symb.append(c[0])

    # velocity
    symb.append(b[0] - b[1])
    symb.append(c[0] - c[1])
    symb.append(p[0] - p[1])

    # acceleration
    symb.append(p[0] - 2.0*p[1] + p[2])

    # jerk
    symb.append(p[0] - 3.0*p[1] + 3.0*p[2] - p[3])
    ''' 

    return np.array(obs)

