from fastapi import FastAPI
from .linear_models import *
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse



#definiendo nombre 
app = FastAPI()


class detrending(BaseModel):
    data_url: str
   
class diference(BaseModel):
    data: list


@app.post("/trend/detrending/linear")
async def link(data: detrending):
    url = data.data_url
    output = primer_endpoint(url) 
    json_compatible_item_data = jsonable_encoder(output)
    return JSONResponse(content=json_compatible_item_data)





@app.post("/trend/difference")
async def link(data: diference):
    data = data.data
    output = segundo_endpoint(data)
    json_compatible_item_data = jsonable_encoder(output)
    return JSONResponse(content=json_compatible_item_data)

