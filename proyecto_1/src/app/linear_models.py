import pandas as pd
import numpy as np 
from sklearn.linear_model import LinearRegression


def dataframe(url_csv,  labels = 'prediction') :
    name = url_csv
    df = pd.read_csv(name)
    df_features = df.drop([labels], axis=1)
    df_label = df[labels]
    return df_features, df_label


def regression(features, label):
    regression = LinearRegression().fit(features, label)
    weight = regression.coef_
    bias = regression.intercept_
    weights = weight.tolist()
    weights.insert(0, bias)
    return  weights


def differencing(label):
    new_serie = []
    for num, lab in enumerate(zip(label)):
        if (num == 0):
            new_serie.append(lab[0])
        else:
            new_serie.append(lab[0]-label[num-1])
    return new_serie


def detrend(model_weights, features, label):
    y_pred =[]
    features = features.values.tolist()
    for i in range(len(features)):
        y_pred.append(model_weights[0] + (model_weights[1] * features[i][0] ))
    label_new = label - y_pred 
    return label_new


def covariance(serie, lag):
    N = np.size(serie)
    mean = np.mean(serie)
    autoCov = 0

    for i in np.arange(0, N-lag + 1):
        autoCov += ((serie[i+lag])-mean)*(serie[i]-mean)

    return autoCov/(N-1)


def primer_endpoint(data_url):
    lag = 51
    sample_acf = []

    features, label = dataframe(data_url)
    model_weights = regression(features, label)
    new_serie = detrend(model_weights, features, label)
    
    for i in range(lag):
        sample_acf.append(covariance(new_serie, i)/ covariance(new_serie, 0))

    output = {"model_weights":[], "sample_acf":[]}
    output["model_weights"] += model_weights
    output["sample_acf"] += sample_acf

    return output




def segundo_endpoint(data):
    lag = 51
    sample_acf = []

    new_serie = differencing(data)
    for i in range(lag):
        sample_acf.append(covariance(new_serie, i)/ covariance(new_serie, 0))
    
    output = {"sample_acf":[]}
    output["sample_acf"] += sample_acf

    return output




'''url = 'https://github.com/ErenCoro/data_linear/blob/main/data.csv?raw=true'
data = [7.42970945517004, -1.72442886840079, 0.507636271398918, 0.099401785447537, 2.80356333327002, -0.205515187062223, 5.33078499411271, 13.4810532502959, 6.08284303373998, 7.02292179144359, 4.7086085017434, 2.86241914763286, 7.4511759170842, 24.9347738199833, 7.63637258210165, 7.62979077881237, 21.0212905108513, 14.3670580153989, 24.2279803479043, 12.132787014697, 19.9117170089851, 26.0739134104548, 21.3669424881042, 29.0340908208122, 31.4035585389661, 21.1939092322177, 27.504843977743, 30.1754284155322, 35.660125785787, 31.374301634563, 26.9584510223971, 20.7917710678645, 33.0357479940199, 35.8480244195971, 40.3464529200624, 41.5668022490189, 24.8857833788973, 28.8703354141134, 27.9052920195993, 43.7698898459073, 40.9334690714617, 41.6105947500503, 36.6452954128365, 48.9354773257329, 54.1666601845099, 56.0043012394246, 42.2421044187755, 59.6728850822704, 41.9278520215718, 55.1159480254006, 50.0521324450239, 57.3874188952443, 65.0784394698322, 52.027900294115, 53.5490907587948, 50.904322166458, 53.5755288815071, 52.5620417219832, 54.1679816306368, 64.1641090421309, 60.3231682437588, 55.9735025965342, 67.6042291764724, 66.5794273051279, 64.3099193761961, 66.3619442901962, 67.8591112814157, 67.9072582218679, 62.4562191707509, 70.262914385865, 79.5033583086278, 62.2747245275228, 75.0571052100106, 77.9131026139576, 75.5585258271807, 69.2739072007765, 71.8307463079608, 72.4935885292941, 78.3518745842609, 75.8760272691402, 81.463456895423, 86.7763672203714, 86.8341956956844, 80.4013911229247, 85.7147402267716, 77.2030039236846, 81.2469507132828, 84.967173238204, 84.7474591223522, 87.3224778514552, 94.7017388025556, 82.0971124569121, 83.2537585946539, 85.1540261686924, 97.3385787327631, 99.9658620112744, 99.051311082952, 93.8786444869056, 105.213293350438, 107.150785058828]   
'''


