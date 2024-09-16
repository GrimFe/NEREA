import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from nerea.reaction_rate import ReactionRate, ReactionRates
from nerea.experimental import Traverse

counts = [0,0,0,0,0,.3,.3,.4,.1,.2,.5,0,.0,1,1,1.5,2,2.5,2,3,3.5,4,4.2,3.8,4.2,3.9,3.9,4.2,4.1,4,
          0,0,0,0,0,.3,.3,.4,.1,.2,.5,0,.0,1,1,1.5,2,2.5,2,3,3.5,4,4.2,3.8,4.2,3.9,3.9,4.2,4.1,4,
          0,0,0,0,0,.3,.3,.4,.1,.2,.5,0,.0,1,1,1.5,2,2.5,2,3,3.5,4,4.2,3.8,4.2,3.9,3.9,4.2,4.1,4,
          3700,4000,4100,4200,3800,3700,3800,3900,4200,
          3900,3900,4200,4100,4000,3700,4000,4100,4200,3800,3700,3800,3900,4200,3900,3900,4200,4100,
          4000,3700,4000,4100,4200,3800,
          3900,3900,4200,4100,4000,3700,4000,4100,4200,3800,3700,3800,3900,4200,3900,3900,4200,4100,
          4000,3700,4000,4100,4200,3800, 3800,3700,3800,3900,4200,
          3900,3900,4200,4100,4000,3700, 3800,3700,3800,3900,4200,
          3900,3900,4200,4100,4000,3700,
          3900,3900,4200,4100,4000,3700,4000,4100,4200,3800,3700,3800,3900,4200,3900,3900,4200,4100,
          4000,3700,4000,4100,4200,3800,3800,3700,3800,3900,4200,
          3900,3900,4200,4100,4000,3700,
          3900,3900,4200,4100,4000,3700,4000,4100,4200,3800,3700,3800,3900,4200,3900,3900,4200,4100,
          4000,3700,4000,4100,4200,3800,3800,3700,3800,3900,4200,
          3900,3900,4200,4100,4000,3700,
          4200,3800, 3900,3900,4200,4100,4000,3700, 4200,3800, 3900,3900,4200,4100,4000,3700, 4200,3800, 3900,3900,4200,4100,4000,3700,
          4200,3800, 3900,3900,4200,4100,3800,3700,3800,3900,4200,
          3900,3900,4200,4100,4000,3700,
          4000,3700, 4200,3800, 3900,3900,4200,4100,4000,3700, 4200,3800, 3900,3900,4200,4100,4000,3700,
          4200,3800, 3900,3900,4200,4100,4000,3700, 4200,3800, 3900,3900,4200,4100,4000,3700, 4200,3800, 3900,3900,4200,4100,4000,3700,
          4100,4000,3700,4000,4100,4200,3800,3700,3800,3900,4200,3900,3900,4200,4100,
          4000,3700,4000,4100,4200,3800, 4200,3800, 3900,3900,4200,4100,4000,3700, 4200,3800, 3900,3900,4200,4100,4000,3700, 4200,3800,
          3900,3900,4200,4100,4000,3700, 4100,4000,3700,4000,4100,4200,3800,3700,3800,3900,4200,3900,3900,4200,4100,
          4000,3700,4000,4100,4200,3800, 4100,4000,3700,4000,4100,4200,3800,3700,3800,3900,4200,3900,3900,4200,4100,
          4000,3700,4000,4100,4200,3800,
          4200,3800, 3900,3900,4200,4100,4000,3700, 4200,3800, 3900,3900,4200,4100,4000,3700, 4200,3800, 3900,3900,4200,4100,4000,3700,
          3.7,3.8,3.9,4,4,4.2,4.1,3.5,3.2,3,2.5,2.2,2,2,1.5,1.5,1.6,1,1,1,
          .5,.6,.4,.3,.5,.3,.5,.6,.1,.3,.2,.1,0,0,0,0,0,0,0,
          0,0,0,0,0,.3,.3,.4,.1,.2,.5,0,.0,1,1,1.5,2,2.5,2,3,3.5,4,4.2,3.8,4.2,3.9,3.9,4.2,4.1,4,
          0,0,0,0,0,.3,.3,.4,.1,.2,.5,0,.0,1,1,1.5,2,2.5,2,3,3.5,4,4.2,3.8,4.2,3.9,3.9,4.2,4.1,4,
          3.7,3.8,3.9,4,4,4.2,4.1,3.5,3.2,3,2.5,2.2,2,2,1.5,1.5,1.6,1,1,1,
          .5,.6,.4,.3,.5,.3,.5,.6,.1,.3,.2,.1,0,0,0,0,0,0,0,
          3.7,3.8,3.9,4,4,4.2,4.1,3.5,3.2,3,2.5,2.2,2,2,1.5,1.5,1.6,1,1,1,
          .5,.6,.4,.3,.5,.3,.5,.6,.1,.3,.2,.1,0,0,0,0,0,0,0,]

@pytest.fixture
def rr1():
    time = [datetime(2024,5,27,13,19,20) + timedelta(seconds=i) for i in range(len(counts))]
    data =  pd.DataFrame({'Time': time, 'value': counts})
    return ReactionRate(data, data.Time.min(),
                        campaign_id='A', experiment_id='B',
                        detector_id=1, deposit_id='dep')

@pytest.fixture
def rr2():
    time = [datetime(2024,5,27,15,12,42) + timedelta(seconds=i) for i in range(len(counts))]
    data = pd.DataFrame({'Time': time, 'value': np.array(counts) / 2})
    return ReactionRate(data, data.Time.min(),
                        campaign_id='A', experiment_id='B',
                        detector_id=1, deposit_id='dep')

@pytest.fixture
def monitor1(rr1):
    data_ = rr1.data.copy()
    data_.value = data_.value.apply(lambda x: 600 if x > 1000 else 1)
    return ReactionRate(data_, data_.Time.min(),
                        campaign_id='A', experiment_id='B',
                        detector_id=2, deposit_id='dep')

@pytest.fixture
def monitor2(rr2):
    data_ = rr2.data.copy()
    data_.value = data_.value.apply(lambda x: 600 if x > 1000 else 1)
    return ReactionRate(data_, data_.Time.min(),
                        campaign_id='A', experiment_id='B',
                        detector_id=2, deposit_id='dep')

@pytest.fixture
def sample_traverse_rr(rr1, rr2):
    return Traverse({'loc A': rr1, 'loc B': rr2})

@pytest.fixture
def sample_traverse_rrs(rr1, monitor1, rr2, monitor2):
    return Traverse({'loc A': ReactionRates({1: rr1, 2: monitor1}),
                     'loc B': ReactionRates({1: rr2, 2: monitor2})})

def test_process(sample_traverse_rr, monitor1, monitor2, sample_traverse_rrs):
    expected_df = pd.DataFrame({'value': [1.        , 0.49878764],
                                'uncertainty': [0.00491095, 0.00215417],
                                'uncertainty [%]': [0.49109513, 0.4318809],
                                'traverse': ['loc A', 'loc B']})
    pd.testing.assert_frame_equal(expected_df, sample_traverse_rr.process([monitor1,
                                                                           monitor2]))
    pd.testing.assert_frame_equal(expected_df, sample_traverse_rrs.process([2, 2]))
