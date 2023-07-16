from django.shortcuts import render

# Create your views here.
import pandas as pd
import numpy as np
import pickle
from rest_framework.response import Response
from rest_framework.decorators import api_view
import sklearn

@api_view(['POST'])
def predict_t20score(request):
    try:
        batting_team=request.data.get('batting_team',None)
        bowling_team=request.data.get('bowling_team',None)
        city=request.data.get('city',None)
        current_score=request.data.get('current_score',None)
        overs=request.data.get('overs',None)
        wickets=request.data.get('wickets',None)
        last_five=request.data.get('last_five',None)
        fields=[batting_team,bowling_team,city,current_score,overs,wickets,last_five]
        if not None in fields:
            overs=float(overs)
            if overs>=5 and overs<=19:
                wickets=float(wickets)
                current_score=float(current_score)
                last_five=float(last_five)
                balls_left=120-(overs*6)
                wickets_left=10-wickets
                crr=current_score/overs
                input=pd.DataFrame([[batting_team,bowling_team,city,current_score,balls_left,wickets_left,crr,last_five]],columns=['batting_team','bowling_team','city','current_score','balls_left','wickets_left','crr','last_five'])
                model_path='model/pipet20men.pkl'
                pipe=pickle.load(open(model_path,'rb'))
                prediction=pipe.predict(input)[0]
                prediction=np.round(prediction,0)
                predictions={
                    'error':'0',
                    'message':'Successfull',
                    'Prediction':prediction
                    }
            else:
                predictions={
                    'error':'0',
                    'message':'Applicable only for 5 to 19 overs'
                    }
            
        else:
            predictions={
                'error':'1',
                'message':'Invalid'
            }
    except Exception as e:
        predictions={
            'error':'1',
            'message':str(e)
        }
    return Response(predictions)


@api_view(['POST'])
def predict_iplscore(request):
    try:
        batting_team=request.data.get('batting_team',None)
        bowling_team=request.data.get('bowling_team',None)
        city=request.data.get('city',None)
        current_score=request.data.get('current_score',None)
        overs=request.data.get('overs',None)
        wickets=request.data.get('wickets',None)
        last_five=request.data.get('last_five',None)
        fields=[batting_team,bowling_team,city,current_score,overs,wickets,last_five]
        if not None in fields:
            overs=float(overs)
            if overs>=5 and overs<=19:
                wickets=float(wickets)
                current_score=float(current_score)
                last_five=float(last_five)
                balls_left=120-(overs*6)
                wickets_left=10-wickets
                crr=current_score/overs
                input=pd.DataFrame([[batting_team,bowling_team,city,current_score,balls_left,wickets_left,crr,last_five]],columns=['batting_team','bowling_team','city','current_score','balls_left','wickets_left','crr','last_five'])
                model_path='model/pipetipl.pkl'
                pipe=pickle.load(open(model_path,'rb'))
                prediction=pipe.predict(input)[0]
                prediction=np.round(prediction,0)
                predictions={
                    'error':'0',
                    'message':'Successfull',
                    'Prediction':prediction
                    }
            else:
                predictions={
                    'error':'0',
                    'message':'Applicable only for 5 to 19 overs'
                    }
        else:
            predictions={
                'error':'1',
                'message':'Invalid'
            }
    except Exception as e:
        predictions={
            'error':'1',
            'message':str(e)
        }
    return Response(predictions)



@api_view(['POST'])
def predict_odiscore(request):
    try:
        batting_team=request.data.get('batting_team',None)
        bowling_team=request.data.get('bowling_team',None)
        city=request.data.get('city',None)
        current_score=request.data.get('current_score',None)
        overs=request.data.get('overs',None)
        wickets=request.data.get('wickets',None)
        last_ten=request.data.get('last_ten',None)
        fields=[batting_team,bowling_team,city,current_score,overs,wickets,last_ten]
        if not None in fields:
            overs=float(overs)
            if overs>=10 and overs<=49:
                wickets=float(wickets)
                current_score=float(current_score)
                last_ten=float(last_ten)
                balls_left=300-(overs*6)
                wickets_left=10-wickets
                crr=current_score/overs
                input=pd.DataFrame([[batting_team,bowling_team,city,current_score,balls_left,wickets_left,crr,last_ten]],columns=['batting_team','bowling_team','city','current_score','balls_left','wickets_left','crr','last_ten'])
                model_path='model/pipeodismen.pkl'
                pipe=pickle.load(open(model_path,'rb'))
                prediction=pipe.predict(input)[0]
                prediction=np.round(prediction,0)
                predictions={
                    'error':'0',
                    'message':'Successfull',
                    'Prediction':prediction
                    }
            else:
                predictions={
                    'error':'0',
                    'message':'Applicable only for 10 to 49 overs'
                    }
        else:
            predictions={
                'error':'1',
                'message':'Invalid'
            }
    except Exception as e:
        predictions={
            'error':'1',
            'message':str(e)
        }
    return Response(predictions)
