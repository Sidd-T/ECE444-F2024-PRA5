import requests
import time
import pytest

# URL of the API endpoint
API_URL = 'http://pra5-new-env.eba-myr9rjmf.us-east-2.elasticbeanstalk.com/predict'
INPUTS = [
    'CIA realizes it has been using black highlighters all these years.',
    'Toyota Recalls 1993 Camry Due To Fact That Owners Really Should Have Bought Something New By Now',
    'The latest climate report warns that global temperatures are rising at an unprecedented rate.',
    'Florida man arrested for trying to get alligator drunk'
]

def test_api_performance():
    # Open a file to write the results
    with open('test_results.txt', 'w') as file:
        for i, input_text in enumerate(INPUTS):
            file.write(f"\nTesting with input {i + 1}: '{input_text}'\n")
            response_times = []

            for j in range(100):
                start = time.time()
                response = requests.post(API_URL, json={'input_text': input_text})
                end = time.time()

                elapsed_time = end - start
                response_times.append(elapsed_time)

                assert response.status_code == 200, f"Error: {response.text}"
                file.write(f"{elapsed_time:.4f}\n")

if __name__ == '__main__':
    pytest.main()
