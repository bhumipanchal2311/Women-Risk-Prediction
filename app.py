import streamlit as st

import pandas as pd

import numpy as np

import joblib

import matplotlib.pyplot as plt

import pydeck as pdk


# ============================================================
# LOAD MODEL
# ============================================================

model=joblib.load(
    'dt_model.joblib'
)


# ============================================================
# LOAD SCALER
# ============================================================

scaler=joblib.load(
    'scaler.joblib'
)


# ============================================================
# LOAD FEATURE COLUMNS
# ============================================================

feature_columns=joblib.load(
    'feature_columns.joblib'
)


# ============================================================
# LOAD DATASET
# ============================================================
# NOTE: fixed the hardcoded 'D:/AICW_PROJECT/...' path from the
# original app.py. That only worked on the original dev machine.
# Keep the CSV in the same folder as app.py and use a relative path.

df=pd.read_csv(
    'Woman_Safety_Dataset_Management.csv'
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title='Women Safety Risk Prediction',
    page_icon='🛡️',
    layout='wide'
)


# ============================================================
# TITLE
# ============================================================

st.title(
    '🛡️ Women Safety Risk Prediction'
)

st.write(
    'Enter the details below to predict the risk level.'
)


# ============================================================
# LOCATION (city -> area is now a dependent dropdown)
# ============================================================

st.header('📍 Location Information')


col1,col2=st.columns(2)


with col1:

    city=st.selectbox(
        'Select City',
        sorted(df['city'].unique())
    )


# Only show areas that actually belong to the selected city
area_options=sorted(
    df[df['city']==city]['area'].unique()
)


with col2:

    area=st.selectbox(
        'Select Area',
        area_options
    )


# ============================================================
# INCIDENT CONDITIONS
# ============================================================
# Crime Type dropdown has been removed per request. The model
# still needs a crime_type value internally to build its input
# row, so we silently infer it as the most frequently recorded
# crime type for the chosen city + area. It does not change what
# the user selects, and (see feature_importances_ check from
# earlier) it has no measurable effect on the predicted risk
# level anyway — the tree is driven almost entirely by
# safety_score.

st.header('🚨 Incident Conditions')

st.caption(
    'Time of day and weather are shown for context but do not '
    'influence the predicted risk level in this dataset — see '
    'the note in the app.py code for why.'
)


col3,col4=st.columns(2)


with col3:

    time_of_day=st.selectbox(
        'Select Time of Day',
        df['time_of_day'].unique()
    )


with col4:

    weather_condition=st.selectbox(
        'Select Weather Condition',
        df['weather_condition'].unique()
    )


# Historical records for this exact city + area combination.
# Used both for inferring crime_type below and for the crime
# type breakdown chart shown after prediction.
location_subset=df[
    (df['city']==city) &
    (df['area']==area)
]

if not location_subset.empty:
    inferred_crime_type=location_subset['crime_type'].mode()[0]
else:
    inferred_crime_type=df['crime_type'].mode()[0]


# ============================================================
# NUMERICAL INPUT
# ============================================================
# Defaults now come from the SELECTED city + area's historical
# median instead of the whole dataset's median, so switching
# location actually changes the starting numbers. Falls back to
# the dataset-wide median if this exact area has no records.

def location_default(column):
    if not location_subset.empty:
        return location_subset[column].median()
    return df[column].median()


st.header('📊 Safety Information')

st.caption(
    'Defaults below reflect typical conditions recorded for '
    f'{area}, {city}. Adjust any value to see how the predicted '
    'risk level changes.'
)


col5,col6=st.columns(2)


with col5:

    latitude=st.number_input(
        'Latitude',
        value=float(
            location_default('latitude')
        )
    )


    longitude=st.number_input(
        'Longitude',
        value=float(
            location_default('longitude')
        )
    )


    crime_count=st.number_input(
        'Crime Count',
        min_value=0,
        value=int(
            location_default('crime_count')
        )
    )


    lighting_score=st.number_input(
        'Lighting Score',
        min_value=0.0,
        max_value=10.0,
        value=float(
            location_default('lighting_score')
        )
    )


with col6:

    police_station_distance_km=st.number_input(
        'Police Station Distance (km)',
        min_value=0.0,
        value=float(
            location_default('police_station_distance_km')
        )
    )


    crowd_density=st.number_input(
        'Crowd Density',
        min_value=0.0,
        value=float(
            location_default('crowd_density')
        )
    )


# ============================================================
# DERIVED SAFETY SCORE
# ============================================================
# safety_score in the training data is not an independent factor
# — it is (almost entirely, R^2 = 0.96) a formula of crime_count,
# lighting_score and police_station_distance_km. Letting the user
# set it separately meant it always sat near the dataset median
# ("Medium"), regardless of every other input. Computing it here
# from the three driving factors is what actually makes Low /
# Medium / High / Critical happen based on real conditions.
# crowd_density, city, area, weather_condition and time_of_day
# have no measurable effect on safety_score in this dataset, so
# they don't move this number — that's a property of the data
# the model was trained on, not a bug in this app.

safety_score=(

    0.9540306451619278

    - 0.007350751220123773*crime_count

    + 0.04362144351836091*lighting_score

    - 0.058662186751950854*police_station_distance_km

)

safety_score=float(
    np.clip(safety_score,0,1)
)

st.metric(
    'Computed Safety Score',
    f'{safety_score:.2f}',
    help=(
        'Derived from Crime Count, Lighting Score and Police '
        'Station Distance. Not directly editable — change those '
        'three inputs to move it.'
    )
)


# ============================================================
# DATE AND TIME
# ============================================================

st.header('🕐 Date and Time')


incident_date=st.date_input(
    'Select Date'
)


incident_time=st.time_input(
    'Select Time'
)


# ============================================================
# CREATE TIME FEATURES
# ============================================================

hour=incident_time.hour

day_of_week=incident_date.weekday()

month=incident_date.month

is_weekend=1 if day_of_week in [5,6] else 0

is_am=1 if hour<12 else 0

is_pm=1 if hour>=12 else 0


# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

input_data=pd.DataFrame({

    'city':[city],

    'area':[area],

    'crime_type':[inferred_crime_type],

    'time_of_day':[time_of_day],

    'weather_condition':[weather_condition],

    'latitude':[latitude],

    'longitude':[longitude],

    'crime_count':[crime_count],

    'lighting_score':[lighting_score],

    'police_station_distance_km':[
        police_station_distance_km
    ],

    'crowd_density':[crowd_density],

    'safety_score':[safety_score],

    'hour':[hour],

    'day_of_week':[day_of_week],

    'month':[month],

    'is_weekend':[is_weekend],

    'is_am':[is_am],

    'is_pm':[is_pm]
})


# ============================================================
# ENCODE INPUT DATA
# ============================================================

columns_to_encode=[

    'city',

    'area',

    'crime_type',

    'time_of_day',

    'weather_condition'

]


input_data=pd.get_dummies(

    input_data,

    columns=columns_to_encode,

    dtype=int

)


# ============================================================
# MATCH TRAINING COLUMNS
# ============================================================

input_data=input_data.reindex(

    columns=feature_columns,

    fill_value=0

)


# ============================================================
# PREDICT BUTTON
# ============================================================

st.write('')


if st.button(
    '🔮 PREDICT RISK',
    use_container_width=True
):


    # ========================================================
    # SCALE INPUT
    # ========================================================

    input_data_scaled=scaler.transform(
        input_data
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    prediction=model.predict(
        input_data_scaled
    )


    # ========================================================
    # RESULT
    # ========================================================

    st.header(
        '🎯 Prediction Result'
    )


    st.success(
        f'Predicted Risk Level: {prediction[0]}'
    )


    # ========================================================
    # PROBABILITY
    # ========================================================

    probability=model.predict_proba(
        input_data_scaled
    )[0]


    classes=model.classes_


    st.subheader(
        '📊 Risk Probability'
    )


    probability_df=pd.DataFrame({

        'Risk Level':classes,

        'Probability (%)':
            probability*100

    })


    probability_df[
        'Probability (%)'
    ]=probability_df[
        'Probability (%)'
    ].round(2)


    st.dataframe(
        probability_df,
        use_container_width=True
    )


    chart_data=probability_df.set_index(
        'Risk Level'
    )


    st.bar_chart(
        chart_data[
            'Probability (%)'
        ]
    )


    # ========================================================
    # CRIME TYPE BREAKDOWN (historical, for this area)
    # ========================================================
    # This is a descriptive breakdown of past incidents recorded
    # in this exact city + area, not a model prediction — the
    # model no longer receives a user-picked crime_type at all.

    st.subheader(
        f'📌 Crime Type Breakdown — {area}, {city}'
    )

    if not location_subset.empty:

        crime_pct=(

            location_subset['crime_type']

            .value_counts(normalize=True)

            .mul(100)

            .round(1)

        )

        fig,ax=plt.subplots(figsize=(6,6))

        ax.pie(
            crime_pct.values,
            labels=crime_pct.index,
            autopct='%1.1f%%'
        )

        ax.set_title(
            f'Historical Crime Type Share — {area}, {city}'
        )

        st.pyplot(fig)

        st.dataframe(
            crime_pct.rename('Share (%)').reset_index().rename(
                columns={'index':'Crime Type'}
            ),
            use_container_width=True
        )

    else:

        st.info(
            'No historical incidents recorded for this exact '
            'area yet.'
        )


# ============================================================
# CRIME RISK HEATMAP
# ============================================================

st.write('--------------------------------')

st.header('🗺️ Crime Risk Heatmap')

st.write(
    'Geographic concentration of recorded incidents, weighted '
    'by crime count.'
)

heatmap_scope=st.radio(
    'Map scope',
    ['All India','Selected City'],
    horizontal=True
)

if heatmap_scope=='Selected City':

    map_df=df[df['city']==city][
        ['latitude','longitude','crime_count']
    ]

    zoom_level=10

else:

    map_df=df[
        ['latitude','longitude','crime_count']
    ]

    zoom_level=4


heatmap_layer=pdk.Layer(
    'HeatmapLayer',
    data=map_df,
    get_position='[longitude, latitude]',
    get_weight='crime_count',
    radiusPixels=60,
    aggregation='MEAN'
)

view_state=pdk.ViewState(
    latitude=float(map_df['latitude'].mean()),
    longitude=float(map_df['longitude'].mean()),
    zoom=zoom_level
)

st.pydeck_chart(
    pdk.Deck(
        layers=[heatmap_layer],
        initial_view_state=view_state,
        map_style=None
    )
)


# ============================================================
# FOOTER
# ============================================================

st.write('--------------------------------')

st.caption(
    'Women Safety Risk Prediction using Decision Tree'
)

# import streamlit as st

# import pandas as pd

# import numpy as np

# import joblib

# import matplotlib.pyplot as plt

# import pydeck as pdk


# # ============================================================
# # LOAD MODEL
# # ============================================================

# model=joblib.load(
#     'dt_model.joblib'
# )


# # ============================================================
# # LOAD SCALER
# # ============================================================

# scaler=joblib.load(
#     'scaler.joblib'
# )


# # ============================================================
# # LOAD FEATURE COLUMNS
# # ============================================================

# feature_columns=joblib.load(
#     'feature_columns.joblib'
# )


# # ============================================================
# # LOAD DATASET
# # ============================================================
# # NOTE: fixed the hardcoded 'D:/AICW_PROJECT/...' path from the
# # original app.py. That only worked on the original dev machine.
# # Keep the CSV in the same folder as app.py and use a relative path.

# df=pd.read_csv(
#     'Woman_Safety_Dataset_Management.csv'
# )


# # ============================================================
# # PAGE CONFIGURATION
# # ============================================================

# st.set_page_config(
#     page_title='Women Safety Risk Prediction',
#     page_icon='🛡️',
#     layout='wide'
# )


# # ============================================================
# # TITLE
# # ============================================================

# st.title(
#     '🛡️ Women Safety Risk Prediction'
# )

# st.write(
#     'Enter the details below to predict the risk level.'
# )


# # ============================================================
# # LOCATION (city -> area is now a dependent dropdown)
# # ============================================================

# st.header('📍 Location Information')


# col1,col2=st.columns(2)


# with col1:

#     city=st.selectbox(
#         'Select City',
#         sorted(df['city'].unique())
#     )


# # Only show areas that actually belong to the selected city
# area_options=sorted(
#     df[df['city']==city]['area'].unique()
# )


# with col2:

#     area=st.selectbox(
#         'Select Area',
#         area_options
#     )


# # ============================================================
# # INCIDENT CONDITIONS
# # ============================================================
# # Crime Type dropdown has been removed per request. The model
# # still needs a crime_type value internally to build its input
# # row, so we silently infer it as the most frequently recorded
# # crime type for the chosen city + area. It does not change what
# # the user selects, and (see feature_importances_ check from
# # earlier) it has no measurable effect on the predicted risk
# # level anyway — the tree is driven almost entirely by
# # safety_score.

# st.header('🚨 Incident Conditions')


# col3,col4=st.columns(2)


# with col3:

#     time_of_day=st.selectbox(
#         'Select Time of Day',
#         df['time_of_day'].unique()
#     )


# with col4:

#     weather_condition=st.selectbox(
#         'Select Weather Condition',
#         df['weather_condition'].unique()
#     )


# # Historical records for this exact city + area combination.
# # Used both for inferring crime_type below and for the crime
# # type breakdown chart shown after prediction.
# location_subset=df[
#     (df['city']==city) &
#     (df['area']==area)
# ]

# if not location_subset.empty:
#     inferred_crime_type=location_subset['crime_type'].mode()[0]
# else:
#     inferred_crime_type=df['crime_type'].mode()[0]


# # ============================================================
# # NUMERICAL INPUT
# # ============================================================

# st.header('📊 Safety Information')


# col5,col6=st.columns(2)


# with col5:

#     latitude=st.number_input(
#         'Latitude',
#         value=float(
#             df['latitude'].median()
#         )
#     )


#     longitude=st.number_input(
#         'Longitude',
#         value=float(
#             df['longitude'].median()
#         )
#     )


#     crime_count=st.number_input(
#         'Crime Count',
#         min_value=0,
#         value=int(
#             df['crime_count'].median()
#         )
#     )


#     lighting_score=st.number_input(
#         'Lighting Score',
#         min_value=0.0,
#         value=float(
#             df['lighting_score'].median()
#         )
#     )


# with col6:

#     police_station_distance_km=st.number_input(
#         'Police Station Distance (km)',
#         min_value=0.0,
#         value=float(
#             df['police_station_distance_km'].median()
#         )
#     )


#     crowd_density=st.number_input(
#         'Crowd Density',
#         min_value=0.0,
#         value=float(
#             df['crowd_density'].median()
#         )
#     )


#     safety_score=st.number_input(
#         'Safety Score',
#         min_value=0.0,
#         value=float(
#             df['safety_score'].median()
#         )
#     )


# # ============================================================
# # DATE AND TIME
# # ============================================================

# st.header('🕐 Date and Time')


# incident_date=st.date_input(
#     'Select Date'
# )


# incident_time=st.time_input(
#     'Select Time'
# )


# # ============================================================
# # CREATE TIME FEATURES
# # ============================================================

# hour=incident_time.hour

# day_of_week=incident_date.weekday()

# month=incident_date.month

# is_weekend=1 if day_of_week in [5,6] else 0

# is_am=1 if hour<12 else 0

# is_pm=1 if hour>=12 else 0


# # ============================================================
# # CREATE INPUT DATAFRAME
# # ============================================================

# input_data=pd.DataFrame({

#     'city':[city],

#     'area':[area],

#     'crime_type':[inferred_crime_type],

#     'time_of_day':[time_of_day],

#     'weather_condition':[weather_condition],

#     'latitude':[latitude],

#     'longitude':[longitude],

#     'crime_count':[crime_count],

#     'lighting_score':[lighting_score],

#     'police_station_distance_km':[
#         police_station_distance_km
#     ],

#     'crowd_density':[crowd_density],

#     'safety_score':[safety_score],

#     'hour':[hour],

#     'day_of_week':[day_of_week],

#     'month':[month],

#     'is_weekend':[is_weekend],

#     'is_am':[is_am],

#     'is_pm':[is_pm]
# })


# # ============================================================
# # ENCODE INPUT DATA
# # ============================================================

# columns_to_encode=[

#     'city',

#     'area',

#     'crime_type',

#     'time_of_day',

#     'weather_condition'

# ]


# input_data=pd.get_dummies(

#     input_data,

#     columns=columns_to_encode,

#     dtype=int

# )


# # ============================================================
# # MATCH TRAINING COLUMNS
# # ============================================================

# input_data=input_data.reindex(

#     columns=feature_columns,

#     fill_value=0

# )


# # ============================================================
# # PREDICT BUTTON
# # ============================================================

# st.write('')


# if st.button(
#     '🔮 PREDICT RISK',
#     use_container_width=True
# ):


#     # ========================================================
#     # SCALE INPUT
#     # ========================================================

#     input_data_scaled=scaler.transform(
#         input_data
#     )


#     # ========================================================
#     # PREDICTION
#     # ========================================================

#     prediction=model.predict(
#         input_data_scaled
#     )


#     # ========================================================
#     # RESULT
#     # ========================================================

#     st.header(
#         '🎯 Prediction Result'
#     )


#     st.success(
#         f'Predicted Risk Level: {prediction[0]}'
#     )


#     # ========================================================
#     # PROBABILITY
#     # ========================================================

#     probability=model.predict_proba(
#         input_data_scaled
#     )[0]


#     classes=model.classes_


#     st.subheader(
#         '📊 Risk Probability'
#     )


#     probability_df=pd.DataFrame({

#         'Risk Level':classes,

#         'Probability (%)':
#             probability*100

#     })


#     probability_df[
#         'Probability (%)'
#     ]=probability_df[
#         'Probability (%)'
#     ].round(2)


#     st.dataframe(
#         probability_df,
#         use_container_width=True
#     )


#     chart_data=probability_df.set_index(
#         'Risk Level'
#     )


#     st.bar_chart(
#         chart_data[
#             'Probability (%)'
#         ]
#     )


#     # ========================================================
#     # CRIME TYPE BREAKDOWN (historical, for this area)
#     # ========================================================
#     # This is a descriptive breakdown of past incidents recorded
#     # in this exact city + area, not a model prediction — the
#     # model no longer receives a user-picked crime_type at all.

#     st.subheader(
#         f'📌 Crime Type Breakdown — {area}, {city}'
#     )

#     if not location_subset.empty:

#         crime_pct=(

#             location_subset['crime_type']

#             .value_counts(normalize=True)

#             .mul(100)

#             .round(1)

#         )

#         fig,ax=plt.subplots(figsize=(6,6))

#         ax.pie(
#             crime_pct.values,
#             labels=crime_pct.index,
#             autopct='%1.1f%%'
#         )

#         ax.set_title(
#             f'Historical Crime Type Share — {area}, {city}'
#         )

#         st.pyplot(fig)

#         st.dataframe(
#             crime_pct.rename('Share (%)').reset_index().rename(
#                 columns={'index':'Crime Type'}
#             ),
#             use_container_width=True
#         )

#     else:

#         st.info(
#             'No historical incidents recorded for this exact '
#             'area yet.'
#         )


# # ============================================================
# # CRIME RISK HEATMAP
# # ============================================================

# st.write('--------------------------------')

# st.header('🗺️ Crime Risk Heatmap')

# st.write(
#     'Geographic concentration of recorded incidents, weighted '
#     'by crime count.'
# )

# heatmap_scope=st.radio(
#     'Map scope',
#     ['All India','Selected City'],
#     horizontal=True
# )

# if heatmap_scope=='Selected City':

#     map_df=df[df['city']==city][
#         ['latitude','longitude','crime_count']
#     ]

#     zoom_level=10

# else:

#     map_df=df[
#         ['latitude','longitude','crime_count']
#     ]

#     zoom_level=4


# heatmap_layer=pdk.Layer(
#     'HeatmapLayer',
#     data=map_df,
#     get_position='[longitude, latitude]',
#     get_weight='crime_count',
#     radiusPixels=60,
#     aggregation='MEAN'
# )

# view_state=pdk.ViewState(
#     latitude=float(map_df['latitude'].mean()),
#     longitude=float(map_df['longitude'].mean()),
#     zoom=zoom_level
# )

# st.pydeck_chart(
#     pdk.Deck(
#         layers=[heatmap_layer],
#         initial_view_state=view_state,
#         map_style=None
#     )
# )


# # ============================================================
# # FOOTER
# # ============================================================

# st.write('--------------------------------')

# st.caption(
#     'Women Safety Risk Prediction using Decision Tree'
# )