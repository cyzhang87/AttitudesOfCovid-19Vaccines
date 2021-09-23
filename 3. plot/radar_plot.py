import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels, fontsize):
            self.set_thetagrids(np.degrees(theta), labels, fontsize=fontsize)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def draw(self, renderer):
            """ Draw. If frame is polygon, make gridlines polygon-shaped """
            if frame == 'polygon':
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer)


        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)


                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

import pandas as pd

font_size = 20
font = {'family': 'Times New Roman',
         'weight': 'normal',
         'size'  : font_size}
plt.rcParams['font.sans-serif'] = 'Times New Roman'
legend_offsets = [(1.3, 1.05), (1.2, 1.2), (1.2, 1.15)]

def radar_subplot(data, ax, legend_offset=(1.3, 1.1)):
    title, case_data = data[0]
    _, labels = data[1]

    ax.set_rgrids([0, 0.2, 0.4, 0.6], fontsize=font_size)
    ax.set_title(title,  position=(0.5, -0.17), fontsize=22)
    for index in range(len(case_data)):
        line = ax.plot(theta, case_data[index], label=labels[index])
        ax.fill(theta, case_data[index], alpha=0.25)
    ax.set_varlabels(emotion_labels, fontsize=font_size)
    ax.legend(frameon=False, loc=1, bbox_to_anchor=legend_offset, prop=font)

vaccine_file = "D:/twitter_data/vaccine_covid_origin_tweets/tweets_analysis_country_state_result_order_new.csv"
vaccine_df = pd.read_csv(vaccine_file)
emotion_labels = ['Fear', 'Joy', 'Anger', 'Disgust', 'Sadness', 'Surprise']
emotion_date =  '2020-12-15'
total_vaccine_df = vaccine_df[vaccine_df['date'] == emotion_date]
person_vaccine_df = total_vaccine_df[total_vaccine_df['org'] == 0]
org_vaccine_df = total_vaccine_df[total_vaccine_df['org'] == 1]

gender_df_list = []
for i in range(2):
    gender_df_list.append(person_vaccine_df[person_vaccine_df['gender'] == i])

age_df_list = []
for i in range(4):
    age_df_list.append(person_vaccine_df[person_vaccine_df['age'] == i])

occ_df_list = []
for i in range(3):
    occ_df_list.append(person_vaccine_df[person_vaccine_df['job_type'] == i])

twi_age_df_list = []
for i in range(3):
    twi_age_df_list.append(person_vaccine_df[person_vaccine_df['twitter_age_class'] == i])

follower_df_list = []
for i in range(3):
    follower_df_list.append(person_vaccine_df[person_vaccine_df['followers_count_class'] == i])

N = len(emotion_labels)
theta = radar_factory(N, frame='polygon')

fig, ax = plt.subplots(2, 3, figsize=(18, 13), subplot_kw=dict(projection='radar'))

# user type
percentages = [[], []]
for j in range(len(emotion_labels)):
    percentages[0].append(sum(person_vaccine_df['ekman'] == emotion_labels[j]) / person_vaccine_df.shape[0])
    percentages[1].append(sum(org_vaccine_df['ekman'] == emotion_labels[j]) / org_vaccine_df.shape[0])
data = [('(a) User Type', [percentages[0], percentages[1]]),
        ('labels', ['Individual', 'Organization'])]
radar_subplot(data, ax[0,0], legend_offset=legend_offsets[0])

# gender
percentages = [[], []]
for j in range(len(emotion_labels)):
    percentages[0].append(sum(gender_df_list[0]['ekman'] == emotion_labels[j]) / gender_df_list[0].shape[0])
    percentages[1].append(sum(gender_df_list[1]['ekman'] == emotion_labels[j]) / gender_df_list[1].shape[0])
data = [('(b) Gender', [percentages[0], percentages[1]]),
        ('labels', ['Male', 'Female'])]
radar_subplot(data, ax[0,1], legend_offset=legend_offsets[0])

# age
percentages = [[], [], [], []]
for j in range(len(emotion_labels)):
    for i in range(len(percentages)):
        percentages[i].append(sum(age_df_list[i]['ekman'] == emotion_labels[j]) / age_df_list[i].shape[0])
data = [('(c) Age', [percentages[0], percentages[1], percentages[2], percentages[3]]),
        ('labels', ['≤18', '19-29', '30-39', '≥40'])]
radar_subplot(data, ax[0,2], legend_offset=legend_offsets[1])

# occupation
percentages = [[], [], []]
for j in range(len(emotion_labels)):
    for i in range(len(percentages)):
        percentages[i].append(sum(occ_df_list[i]['ekman'] == emotion_labels[j]) / occ_df_list[i].shape[0])
data = [('(d) Occupation', [percentages[0], percentages[1], percentages[2]]),
        ('labels', ['Type 1', 'Type 2', 'Type 3'])]
radar_subplot(data, ax[1,0], legend_offset=legend_offsets[2])

# Twitter Age
percentages = [[], [], []]
for j in range(len(emotion_labels)):
    for i in range(len(percentages)):
        percentages[i].append(sum(twi_age_df_list[i]['ekman'] == emotion_labels[j]) / twi_age_df_list[i].shape[0])
data = [('(e) Twitter Age', [percentages[0], percentages[1], percentages[2]]),
        ('labels', ['<5', '10-15', '≥15'])]
radar_subplot(data, ax[1,1], legend_offset=legend_offsets[2])

# Follower Count
percentages = [[], [], []]
for j in range(len(emotion_labels)):
    for i in range(len(percentages)):
        percentages[i].append(sum(follower_df_list[i]['ekman'] == emotion_labels[j]) / follower_df_list[i].shape[0])
data = [('(f) Follower Count', [percentages[0], percentages[1], percentages[2]]),
        ('labels', ['<500', '500-5000', '≥5000'])]
radar_subplot(data, ax[1,2], legend_offset=legend_offsets[2])

plt.subplots_adjust(wspace=0.2, hspace=0.1)
plt.savefig("radar_0.pdf", bbox_inches='tight')

plt.show()