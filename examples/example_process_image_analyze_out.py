import os
import pandas as pd
import matplotlib

matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import seaborn as sns



save_dir = os.path.join(os.path.expanduser('~'), 'Pictures', '2025_08_08_Nextracker_10287', 'processed')
filename = os.path.join(save_dir, '00_processed_data.csv')
df = pd.read_csv(filename)

print(df.keys())
# Index(['filename', 'Stress Step', 'Module', 'Location', 'SWPR Mean (%)',
#        'SWPR Median (%)', 'Thickness Mean (nm)', 'Thickness Median (nm)',
#        'Porosity Mean (%)', 'Porosity Median (%)', 'SWPR Std (%)',
#        'Thickness Std (nm)', 'Porosity Std (%)'],

for k in range(len(df)):
    is_fixed_loc = df.loc[k,'Location'] in ['P1', 'P2', 'P3', 'P4']
    if is_fixed_loc:
        df.loc[k, 'Location-Type'] = 'Fixed Location'
    else:
        df.loc[k, 'Location-Type'] = 'Track Location'
df['Stress - Module - Location-Type'] = df['Stress Step'].astype(str) + ' - ' + df['Module'] + ' - ' + df['Location-Type']
df['Stress - Module - Location'] = df['Stress Step'].astype(str) + ' - ' + df['Module'] + ' - ' + df['Location']



#  --------- SWPR Mean (%) by Stress Step, Module, Fixed Location (P1-P4) ---------

categories = df['Stress - Module - Location-Type'].unique()
categories = [cat for cat in categories if 'Fixed Location' in cat]
order = sorted(categories, key=lambda x: (0 if 'pre' in x.lower() else 2 if 'post' in x.lower() else 1))

mean_swpr_fixed_loc_pre = df[df['Stress Step'].str.contains('Pre') & (df['Location-Type'] == 'Fixed Location')]['SWPR Mean (%)'].mean()
mean_swpr_fixed_loc_post = df[df['Stress Step'].str.contains('Post') & (df['Location-Type'] == 'Fixed Location')]['SWPR Mean (%)'].mean()

sns.set(style='darkgrid')
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(
    x='Stress - Module - Location-Type',
    y='SWPR Mean (%)',
    hue='Module',
    data=df[df['Location-Type']=='Fixed Location'],
    ax=ax,
    palette='Set2',
    fliersize=5,
    linewidth=1.5,
    order=order,
)
plt.tight_layout()
plt.xticks(rotation=90)
plt.text(0.5,3,
         f'Pre-Stress Mean SWPR: {mean_swpr_fixed_loc_pre:.2f} %\n'
         f'Post-Stress Mean SWPR: {mean_swpr_fixed_loc_post:.2f} %\n'
         f'Delta SWPR: {mean_swpr_fixed_loc_post - mean_swpr_fixed_loc_pre:.2f} %'

         )
# turn off legend
if ax.get_legend() is not None:
    ax.get_legend().remove()
plt.title('SWPR Mean (%) by Stress Step, Module, Fixed Location (P1-P4)')
plt.xlabel('Stress Step - Module - Location Type')
plt.ylabel('SWPR Mean (%)')
plt.ylim([0,4])
# plt.legend(title='Stress Step - Module - Location Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig(os.path.join(save_dir, 'swpr_mean_by_stress_module_fixed.png'), dpi=300, bbox_inches='tight')
plt.show()





#  --------- SWPR Mean (%) by Stress Step, Module, Fixed Location (P1-P4) ---------

categories = df['Stress - Module - Location-Type'].unique()
# categories = [cat for cat in categories if 'Fixed Location' in cat]
order = sorted(categories, key=lambda x: (0 if 'pre' in x.lower() else 2 if ('post' in x.lower() and 'track' in x.lower()) else 1))

# mean_swpr_fixed_loc_pre = df[df['Stress Step'].str.contains('Pre') & (df['Location-Type'] == 'Fixed Location')]['SWPR Mean (%)'].mean()
# mean_swpr_fixed_loc_post = df[df['Stress Step'].str.contains('Post') & (df['Location-Type'] == 'Fixed Location')]['SWPR Mean (%)'].mean()

sns.set(style='darkgrid')
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(
    x='Stress - Module - Location-Type',
    y='SWPR Mean (%)',
    hue='Module',
    data=df,
    ax=ax,
    palette='Set2',
    fliersize=5,
    linewidth=1.5,
    order=order,
)
plt.tight_layout()
plt.xticks(rotation=90)
# plt.text(0.5,3,
#          f'Pre-Stress Mean SWPR: {mean_swpr_fixed_loc_pre:.2f} %\n'
#          f'Post-Stress Mean SWPR: {mean_swpr_fixed_loc_post:.2f} %\n'
#          f'Delta SWPR: {mean_swpr_fixed_loc_post - mean_swpr_fixed_loc_pre:.2f} %'
#
#          )
# turn off legend
if ax.get_legend() is not None:
    ax.get_legend().remove()
plt.title('SWPR Mean (%) by Stress Step, Module, Location Type')
plt.xlabel('Stress Step - Module - Location Type')
plt.ylabel('SWPR Mean (%)')
plt.ylim([0,4])
# plt.legend(title='Stress Step - Module - Location Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig(os.path.join(save_dir, 'swpr_mean_by_stress_module_location_type.png'), dpi=300, bbox_inches='tight')
plt.show()




# --- P1 only, definitely not in a track.
categories = df['Stress - Module - Location'].unique()
categories = [cat for cat in categories if 'P1' in cat]
order = sorted(categories, key=lambda x: (0 if 'pre' in x.lower() else 2 if 'post' in x.lower() else 1))


sns.set(style='darkgrid')
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(
    x='Stress - Module - Location-Type',
    y='SWPR Mean (%)',
    hue='Module',
    data=df[df['Location']=='P1'],
    ax=ax,
    palette='Set2',
    fliersize=5,
    linewidth=1.5,
    # order=order,
)
plt.tight_layout()
plt.xticks(rotation=90)
# plt.text(0.5,3,
#          f'Pre-Stress Mean SWPR: {mean_swpr_fixed_loc_pre:.2f} %\n'
#          f'Post-Stress Mean SWPR: {mean_swpr_fixed_loc_post:.2f} %\n'
#          f'Delta SWPR: {mean_swpr_fixed_loc_post - mean_swpr_fixed_loc_pre:.2f} %'
#
#          )
# turn off legend
if ax.get_legend() is not None:
    ax.get_legend().remove()
plt.title('SWPR Mean (%) by Stress Step, Module, Fixed Location (P1)')
plt.xlabel('Stress Step - Module - Location Type')
plt.ylabel('SWPR Mean (%)')
plt.ylim([0,4])
# plt.legend(title='Stress Step - Module - Location Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig(os.path.join(save_dir, 'swpr_mean_by_stress_module_P1.png'), dpi=300, bbox_inches='tight')
plt.show()




# For each BOM:


#  --------- SWPR Mean (%) by Stress Step, Module, Fixed Location (P1-P4) ---------

mod_numbers= [[1,6],
              [7,12],
                [13,18],
                ]

df['Module Number'] = df['Module'].str.extract(r'(\d+)').astype(int)
for mod_number in mod_numbers:
    df_curr = df[df['Module Number'].between(mod_number[0], mod_number[1])]
    categories = df_curr['Stress - Module - Location-Type'].unique()
    # categories = [cat for cat in categories if 'Fixed Location' in cat]
    order = sorted(categories, key=lambda x: (0 if 'pre' in x.lower() else 2 if ('post' in x.lower() and 'track' in x.lower()) else 1))

    # mean_swpr_fixed_loc_pre = df[df['Stress Step'].str.contains('Pre') & (df['Location-Type'] == 'Fixed Location')]['SWPR Mean (%)'].mean()
    # mean_swpr_fixed_loc_post = df[df['Stress Step'].str.contains('Post') & (df['Location-Type'] == 'Fixed Location')]['SWPR Mean (%)'].mean()

    sns.set(style='darkgrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        x='Stress - Module - Location-Type',
        y='SWPR Mean (%)',
        hue='Module',
        data=df_curr,
        ax=ax,
        palette='Set2',
        fliersize=5,
        linewidth=1.5,
        order=order,
    )
    plt.tight_layout()
    plt.xticks(rotation=90)
    # plt.text(0.5,3,
    #          f'Pre-Stress Mean SWPR: {mean_swpr_fixed_loc_pre:.2f} %\n'
    #          f'Post-Stress Mean SWPR: {mean_swpr_fixed_loc_post:.2f} %\n'
    #          f'Delta SWPR: {mean_swpr_fixed_loc_post - mean_swpr_fixed_loc_pre:.2f} %'
    #
    #          )
    # turn off legend
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    plt.title('SWPR Mean (%) by Stress Step, Module, Location Type')
    plt.xlabel('Stress Step - Module - Location Type')
    plt.ylabel('SWPR Mean (%)')
    plt.ylim([0,4])
    # plt.legend(title='Stress Step - Module - Location Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(os.path.join(save_dir, 'swpr_mean_by_stress_module_{}.png'.format(mod_number[0])), dpi=300, bbox_inches='tight')
    plt.show()
