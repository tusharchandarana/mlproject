from app import app

client = app.test_client()
response = client.post(
    '/predictdata',
    data={
        'gender': 'female',
        'race_ethnicity': 'group A',
        'parental_level_of_education': "bachelor's degree",
        'lunch': 'standard',
        'test_preparation_course': 'completed',
        'reading_score': '85',
        'writing_score': '90',
    }
)
print('status:', response.status_code)
print(response.get_data(as_text=True)[:500])
