# Lambda Functions 


#Lambda Fucntion for Serializing the image data 

import json
import sagemaker
import base64
from sagemaker.serializers import IdentitySerializer

# Fill this in with the name of your deployed model
ENDPOINT = "image-classification-2023-10-27-09-47-27-575" 

def lambda_handler(event, context):

    # Decode the image data
    image = base64.b64decode(event['body']['image_data'])

    # Instantiate a Predictor
    predictor = sagemaker.predictor.Predictor(ENDPOINT)

    # For this model the IdentitySerializer needs to be "image/png"
    predictor.serializer = IdentitySerializer("image/png")
    
    # Make a prediction:
    inferences = predictor.predict(image)
    
    # We return the data back to the Step Function    
    event["body"]["inferences"] = json.loads(inferences)
    return {
        'statusCode': 200,
        'body': event["body"]
    }


#Lambda Function for making predictions 
import json
import sagemaker
import base64
from sagemaker.serializers import IdentitySerializer

# Fill this in with the name of your deployed model
ENDPOINT = "image-classification-2023-10-27-09-47-27-575" 

def lambda_handler(event, context):

    # Decode the image data
    image = base64.b64decode(event['body']['image_data'])## TODO: fill in)

    # Instantiate a Predictor
    predictor = sagemaker.predictor.Predictor(ENDPOINT)## TODO: fill in

    # For this model the IdentitySerializer needs to be "image/png"
    predictor.serializer = IdentitySerializer("image/png")
    
    # Make a prediction:
    inferences = predictor.predict(image) ## TODO: fill in
    
    # We return the data back to the Step Function    
    event["body"]["inferences"] = json.loads(inferences)
    return {
        'statusCode': 200,
        'body': event["body"]
    }





#Lambda function for setting the threshold after inference

import json


THRESHOLD = .93


def lambda_handler(event, context):
    
    # Grab the inferences from the event
    inferences = event['body']['inferences']
    
    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = any([x > THRESHOLD for x in inferences])
    
    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise(Exception("THRESHOLD_CONFIDENCE_NOT_MET"))

    return {
        'statusCode': 200,
        'body': event["body"]
    }
