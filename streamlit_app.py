import numpy as np
import altair as alt
import pandas as pd 
from datetime import datetime
import datetime as dt
import pickle
import streamlit as st
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
from random import randint

#########################################
#Proyecto Dolar Blue
#########################################
today = datetime(2022,11,4)
if datetime.now().day == 4:
    today = datetime.now()
df = pd.DataFrame(columns=['Año', 'Semana', 'Mes', 'Dia_mes', 'Dia_semana'])
for i in range(0,30):
    date = today+dt.timedelta(days=i+1)
    date_as_list = [date.year,date.isocalendar()[1],date.month,date.day,date.isocalendar()[2]]
    df.loc[i] = date_as_list
#########################################
#Modelo 1
#########################################
X_cols = ['Año', 'Semana', 'Mes', 'Dia_mes', 'Dia_semana']
y_cols = ['valor_dolar_oficial', 'valor_merval', 'valor_circulacion_monetaria', 'valor_base_monetaria', 'valor_efec_ent_financieras']
model = pickle.load(open(r'Dolar_Blue/Modelos/Dolar Blue/Modelo 1/model.pkl','rb'))
X = df[X_cols].values
y_pred = model.predict(X)
df_X = pd.DataFrame(X,columns=X_cols)
positive_columns = [value for value in y_cols if value.__contains__('valor')]
df_ypred = pd.DataFrame(y_pred,columns=y_cols)
df_ypred[positive_columns] = df_ypred[positive_columns].abs()
df_modelo1 = pd.merge(df_X, df_ypred, left_index=True, right_index=True)
#########################################
#Modelo 2
#########################################
X_cols = ['Año', 'Semana', 'Mes', 'Dia_mes', 'Dia_semana']
y_cols = ['valor_dolar_blue']
model = pickle.load(open(r'Dolar_Blue/Modelos/Dolar Blue/Modelo 2/model.pkl','rb'))
X = df[X_cols].values
y_pred = model.predict(X)
df_X = pd.DataFrame(X,columns=X_cols)
positive_columns = [value for value in y_cols if value.__contains__('valor')]
df_ypred = pd.DataFrame(y_pred,columns=y_cols)
df_ypred[positive_columns] = df_ypred[positive_columns].abs()
df_modelo2 = pd.merge(df_X, df_ypred, left_index=True, right_index=True)
#########################################
#Modelo 1
#########################################
X_cols = ['valor_dolar_oficial', 'valor_merval', 'valor_circulacion_monetaria', 'valor_base_monetaria', 'valor_efec_ent_financieras']
y_cols = ['valor_dolar_blue']
model = pickle.load(open(r'Dolar_Blue/Modelos/Dolar Blue/Modelo 3/model.pkl','rb'))
X = df_modelo1[X_cols].values
y_pred = model.predict(X)
df_X = pd.DataFrame(X,columns=X_cols)
positive_columns = ['valor_dolar_blue']
df_ypred = pd.DataFrame(y_pred,columns=['valor_dolar_blue'])
df_ypred[positive_columns] = df_ypred[positive_columns].abs()
df_modelo3 = pd.merge(df_X, df_ypred, left_index=True, right_index=True)
df_modelo3 = df_modelo1.merge(df_modelo3,how='left')[['Año', 'Semana', 'Mes', 'Dia_mes', 'Dia_semana','valor_dolar_blue']]
#########################################
#DataFrame Final
#########################################
predicciones = df[df.columns[0:5]].merge(df_modelo2,how='left').rename(columns={'valor_dolar_blue':'pred1_blue'})
predicciones = predicciones.merge(df_modelo3,how='left').rename(columns={'valor_dolar_blue':'pred2_blue'}).dropna()
predicciones['pred_diff'] = (predicciones['pred2_blue']-predicciones['pred1_blue'])
predicciones['pred_diff'] = predicciones['pred_diff'].abs()
predicciones = predicciones[list(predicciones.columns[0:6])+list(predicciones.columns[7:])+list(predicciones.columns[6:7])]
predicciones['pred_promedio'] = (predicciones['pred2_blue']+predicciones['pred1_blue'])/2
predicciones['pred_promedio'] = predicciones['pred_promedio']-4
predicciones['Fecha'] = predicciones['Año'].astype(str)+'-'+predicciones['Mes'].astype(str)+'-'+predicciones['Dia_mes'].astype(str)
predicciones = predicciones[['Fecha','pred1_blue','pred2_blue','pred_promedio']].rename(columns={'pred1_blue':'Dolar Blue Minimo','pred2_blue':'Dolar Blue Maximo','pred_promedio':'Dolar Blue Esperado'})
predicciones.to_csv('Dolar_Blue/predicciones_{}.csv'.format(today.strftime('%Y_%m_%d')),index=False)
#########################################
#Chart
#########################################
predicciones_ct = pd.DataFrame(columns=['Fecha','Valor','Dolar Blue'])
for index in predicciones.index:
    predicciones_ct.loc[len(predicciones_ct)] = [predicciones.loc[index,'Fecha'],round(predicciones.loc[index,'Dolar Blue Minimo'],2),'Valor Minimo']
    predicciones_ct.loc[len(predicciones_ct)] = [predicciones.loc[index,'Fecha'],round(predicciones.loc[index,'Dolar Blue Esperado'],2),'Valor Esperado']
    predicciones_ct.loc[len(predicciones_ct)] = [predicciones.loc[index,'Fecha'],round(predicciones.loc[index,'Dolar Blue Maximo'],2),'Valor Maximo']
def get_chart(data):
    hover = alt.selection_single(
        fields=["Fecha"],
        nearest=True,
        on="mouseover",
        empty="none",
    )
    lines = (
        alt.Chart(data, title="Prediccion del Dolar Blue a 30 Dias  ")
        .mark_line()
        .encode(
            x="Fecha",
            y="Valor",
            color="Dolar Blue",
        )
    )
    # Draw points on the line, and highlight based on selection
    points = lines.transform_filter(hover).mark_circle(size=65)
    # Draw a rule at the location of the selection
    tooltips = (
        alt.Chart(data)
        .mark_rule()
        .encode(
            x="yearmonthdate(Fecha)",
            y="Valor",
            opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
            tooltip=[
                alt.Tooltip("yearmonthdate(Fecha)", title="Fecha"),
                alt.Tooltip("Valor", title="Valor (ARS)"),
                alt.Tooltip("Dolar Blue", title="Prediccion"),
            ],
        )
        .add_selection(hover)
    )
    return (lines + points + tooltips).interactive()
chart = get_chart(predicciones_ct)

#########################################
# 
#########################################    
with st.sidebar:
    selected = option_menu(
        menu_title = None,
        options = ['Proyectos','Contacto'],
        icons = ['book','envelope'],
        key='Navigation'
    )


    
if selected == 'Proyectos':
    #with st.sidebar:
    #    components.html("""<script src="https://platform.linkedin.com/badges/js/profile.js" async defer type="text/javascript"></script>
    #        <div class="badge-base LI-profile-badge" data-locale="en_US" data-size="large" data-theme="dark" data-type="VERTICAL" data-vanity="thiago-ferster-924a0924a" data-version="v1"><a class="badge-base__link LI-simple-link" href="https://ar.linkedin.com/in/thiago-ferster-924a0924a?trk=profile-badge"></a></div>"""
    #        ,height=810)
    selected_proyecto = st.selectbox('',('Dolar Blue - Machine Learning', 'Auto-ETL - Data Engineering'))

#########################################
# Proyecto Dolar Blue
#########################################    
    if selected_proyecto == 'Dolar Blue - Machine Learning':

        selected_blue = option_menu(
        menu_title = None,
        options = ['Inicio','Codigo', 'Grafico'],
        icons = ['currency-dollar','code-square','graph-up'],
        key='Proyectos',
        orientation='horizontal'
        )
        
#########################################
# Blue Inicio
#########################################    

        if selected_blue == "Inicio":
            st.markdown("""
            # Introduccion
            #### Este es un proyecto con fines didacticos para poner en practica las habilidades adquiridas en el bootcamp de Henry y cursos de Platzi realizados en torno al Machine Learning y Feature Engineering.

            # Propuesta
            - #### Obtener datos financieros de la API del BCRA.
            - #### Procesarlos para ponerlos en la produccion de varios modelos de Machine Learning.
            - #### Generar una aproximacion futura del Dolar Blue mediante los modelos creados.

            # Procesos
            ### Ingesta de los Datos
            - #### Se realiza una serie de consultas a la API del Banco Central de la Republica Argentina para obtener datos financieros, se identifica cuales de esos datos son los mas linealmente predecibles.
            """)

            st.image(r'_src/assets/Diagrama_1_DB.png')

            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')

            st.markdown("""
            
            ### Puesta en Produccion de los Modelos
            - #### Luego de tratados los datos e identificados los mejores valores de los que se puede obtener una prediccion lineal, se entrenan los modelos.
            - #### El primer modelo recibira la fecha y devolvera una prediccion de los valores financieros seleccionados, a excepcion del dolar blue.
            - #### El segundo modelo recibira la fecha y devolvera la prediccion del dolar blue, mas estable y por debajo de la realidad.
            - #### El tercer modelo recibira los valores del primer modelo y devolvera la predicion del dolar blue, con los pesos ajustados para predecir por arriba y de manera mas volatil
            - #### Con los dos valores que se obtuvo del Dolar Blue se obtendra el valor esperado en base a un promedio y un ajuste.
            """)
            st.image(r'_src/assets/Diagrama_2_DB.png')

#########################################
# Blue - Codigo
#########################################    

        if selected_blue == "Codigo":
            st.write('## Funciones utilizadas')

            st.write("""
            ##### Funciones para consulta, cortesia de mi TA en Henry,  [Alfredo B. Trujillo](https://www.linkedin.com/in/alfredobtrujillop/).
            """)

            st.code("""
            def download_data(url):
    token = 'ey2hbGciOiJIUzUxMiIsI3R5cCIfIkpXVCJ9.eyJleHAiOjE2OTc0NTA1NzUsInR5cGUiUeJleHRccm5hbCSsInHzZXIiOiJjb2Rla292YUBnbWFpbC5jb20ifQ.efI8Sujs8ew_LxCMyRwuuaaQ3-uRdwfHZoH-2s5BSLAW_yCWF4F_RzbTRzv5Q0xI4l3Yk7Maruyk0REeBXI75Q'
    headers = {'Authorization': 'BEARER '+token}
    response = requests.get(url, headers= headers)
    status = response.status_code
    raw_data = response.json()
    if status != 200:
        return print('Error en la url ingresada')
    else:
        return raw_data

            """)

            st.code("""
            def transform_json_to_DF(raw_data,day):
    now = dt.datetime.today()
    day_365 = now -dt.timedelta(days = day+1)
    date = []
    values = []
    for i in raw_data:
        if i['d'] <= now.strftime('%Y-%m-%d') and i['d'] >= day_365.strftime('%Y-%m-%d'):
            date.append(i['d'])
            values.append(i['v'])
        else:
            continue
    data = pd.DataFrame({'Fecha': date, 'Valor': values})
    return data
            """)


            st.write('## Ingesta y tratamiento de los datos')

            st.write("""
            ##### Definimos un Diccionario para cada consulta.
            """)

            st.code("""
            data_dict = {
    'dolar_oficial' : download_data('https://api.estadisticasbcra.com/usd_of'),
    'dolar_blue' : download_data('https://api.estadisticasbcra.com/usd'),
    'merval' : download_data('https://api.estadisticasbcra.com/merval'),
    'merval_usd' : download_data('https://api.estadisticasbcra.com/merval_usd'),
    'reservas' : download_data('https://api.estadisticasbcra.com/reservas'),
    'circulacion_monetaria' : download_data('https://api.estadisticasbcra.com/circulacion_monetaria'),
    'base_monetaria' : download_data('https://api.estadisticasbcra.com/base'),
    'base_monetaria_usd_blue' : download_data('https://api.estadisticasbcra.com/base_usd'),
    'efec_ent_financieras' : download_data('https://api.estadisticasbcra.com/efectivo_en_ent_fin')
}
                    """)

            st.write("""
            ##### Se crea un diccionario con cada consulta en forma de DataFrame para cada clave, luego se renombran los valores dentro de cada DataFrame, usando Dicctionary Comprehension.
            """)

            st.code("""
            df_dict = {'df_{}'.format(key):transform_json_to_DF(value,1095) for (key,value) in data_dict.items()}
df_dict = {key:value.rename(columns={'Valor': 'valor_{}'.format(key[3:])}, inplace= False) for (key,value) in df_dict.items()}
                    """)

            st.write('##### Se crea un DataFrame general con todos los datos recolectados.')

            st.code("""
            df = pd.DataFrame(df_dict['df_dolar_oficial']['Fecha'])
for key in list(df_dict.keys()):
    df = df.merge(df_dict[key],how='left')
                    """)


            st.write('##### Se define un DataFrame con la variacion porcentual de cada dato recolectado')

            st.code("""
            cols = [value for value in list(df.columns) if value not in ['Fecha']]
cols = [value for value in list(df.columns) if value not in ['Fecha']]
df_variacion_porcentual = df.copy()
df_variacion_porcentual[cols] = df_variacion_porcentual[cols].apply(lambda x: x.pct_change())
df_variacion_porcentual = df_variacion_porcentual.rename(columns=lambda x: x.replace('valor','var(%)'))              
                    """)

            st.write('##### Luego uno con la Variacion Neta')

            st.code("""
            cols = [value for value in list(df.columns) if value not in ['Fecha']]
df_variacion = df.copy()
df_variacion[cols] = df_variacion[cols]-df_variacion[cols].shift(1)
df_variacion = df_variacion.rename(columns=lambda x: x.replace('valor','var($)'))
                    """)

            st.write('##### Unimos los Datos originales con su variacion porcentual, respectivamente organizados como | Valor | Valor_Val_Net | Valor_Var_% |')

            st.code("""
            cols = [value for value in list(df.columns) if value not in ['Fecha']]
colsvarp = [value for value in list(df_variacion_porcentual.columns) if value not in ['Fecha']]
colschange = [value for value in list(df_variacion.columns) if value not in ['Fecha']]
columns_ziplist  = list(zip(cols,colschange,colsvarp))
cols3 = [item for nested_items in columns_ziplist for item in nested_items]
cols3.insert(0,'Fecha')
datos = df.merge(df_variacion_porcentual,how='left').merge(df_variacion,how='left')[cols3].dropna()
                    """)

            st.write('##### Se convierte la fecha a valores numericos')

            st.code("""
            cols = [value for value in list(datos.columns) if value not in ['Fecha']]
datos.Fecha.apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))
cols.insert(0,'Dia_semana')
cols.insert(0,'Dia_mes')
cols.insert(0,'Mes')
cols.insert(0,'Semana')
cols.insert(0,'Año')
datos['Año'] = datos.Fecha.apply(lambda x: datetime.strptime(x,'%Y-%m-%d').year)
datos['Semana'] = datos.Fecha.apply(lambda x: datetime.strptime(x,'%Y-%m-%d').isocalendar()[1])
datos['Mes'] = datos.Fecha.apply(lambda x: datetime.strptime(x,'%Y-%m-%d').month)
datos['Dia_mes'] = datos.Fecha.apply(lambda x: datetime.strptime(x,'%Y-%m-%d').day)
datos['Dia_semana'] = datos.Fecha.apply(lambda x: datetime.strptime(x,'%Y-%m-%d').isocalendar()[2])
datos = datos[cols]
                    """)

            st.write("""
            ##### DataFrame resultante""")

            datos_df = pd.read_csv(r'Dolar_Blue/datos.csv')
            datos_index = st.slider(
                '',
                0,len(datos_df),(len(datos_df)//2,len(datos_df))
            )
            st.dataframe(datos_df.iloc[datos_index[0]:datos_index[1],:])

            st.write("""
            ## Testeo y exportacion de los Modelos""")

            st.write('#### Modelo 1')

            st.write('##### Definimos las columnas para el modelo, las pasamos a array de NumPy y dividimos entre train y test.')

            st.code("""
            cols = list(datos.columns)
X_cols = list(datos.columns[0:5])
y_cols = [value for value in list(datos.columns[5:]) if value != 'valor_dolar_blue']
X = datos[X_cols].values
y = datos[y_cols].values
X_train , X_test , y_train , y_test = train_test_split(X,y)
""")

            st.write('##### Entrenamos el modelo MultiOutputRegressor y una pipeline para escalar los datos, y predecimos los datos de testeo.')
            st.write('MultiOutputRegressor: Usa las columnas de X para predecir una por una las columnas de Y, sin tomarlas como un conjunto.')

            st.code("""
            model = MultiOutputRegressor(Ridge(random_state=100))
model = make_pipeline(StandardScaler(),model)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
            """)

            st.write('##### Definimos un DataFrame para los datos de Prueba, convirtiendo las columnas que no pueden ser negativas a valor absoluto(positivas).')

            st.code("""
            df_ytest = pd.DataFrame(y_test,columns=y_cols)
positive_cols = [value for value in y_cols if value.__contains__('valor')]
df_ypred = pd.DataFrame(y_pred,columns=y_cols)
df_ypred[positive_cols] = df_ypred[positive_cols].abs()
            """)

            st.write('##### Testeamos las columnas que mejor se pudieron predecir para usarlas en la produccion')

            st.code("""
            for column in df_ypred.columns:
    r2 = r2_score(np.array(df_ytest[column]),np.array(df_ypred[column]))
    MSE = mean_squared_error(np.array(df_ytest[column]),np.array(df_ypred[column]))
    RMSE = mean_squared_error(np.array(df_ytest[column]),np.array(df_ypred[column]),squared= False)
    if(r2>=0.75):
        print('***********************')
        print(' R2 Score for {}: {}'.format(column,r2))
        print(' MSE for {}: {}'.format(column,MSE))
        print(' RMSE for {}: {}'.format(column,RMSE))
        print('***********************')
            """)

            st.write("""
            ***********************
 R2 Score for valor_dolar_oficial: 0.9460628660599424]\n
 MSE for valor_dolar_oficial: 29.369944570518076\n
 RMSE for valor_dolar_oficial: 5.419404447955335\n
***********************
***********************
 R2 Score for valor_merval: 0.8593805981848737\n
 MSE for valor_merval: 120739205.97918582\n
 RMSE for valor_merval: 10988.139331988188\n
***********************
***********************
 R2 Score for valor_circulacion_monetaria: 0.9638650111386968\n
 MSE for valor_circulacion_monetaria: 14746873908.927849\n
 RMSE for valor_circulacion_monetaria: 121436.70741965894\n
***********************
***********************
 R2 Score for valor_base_monetaria: 0.9400037253725954\n
 MSE for valor_base_monetaria: 35531395672.90041\n
 RMSE for valor_base_monetaria: 188497.73386675082\n
***********************
***********************
 R2 Score for valor_efec_ent_financieras: 0.8201371338188644\n
 MSE for valor_efec_ent_financieras: 550927429.6955314\n
 RMSE for valor_efec_ent_financieras: 23471.843338253846
***********************
            """)

            st.write('##### Identificadas las variables creamos y exportamos el modelo.')

            st.code("""
            X_cols = list(datos.columns[0:5])
y_cols = [value for value in list(datos.columns[5:]) if value in ['valor_dolar_oficial','valor_merval','valor_circulacion_monetaria','valor_base_monetaria','valor_efec_ent_financieras']]
model = MultiOutputRegressor(Ridge(random_state=50))
model = make_pipeline(StandardScaler(),model)
model.fit(X,y)
y_pred = model.predict(X)
dump(model, r'Modelos/Dolar Blue/Modelo 1/model.joblib')
            """)

            st.write('##### Creamos un DataFrame con el Primer Modelo.')

            st.code("""
            df_X = pd.DataFrame(X,columns=X_cols)
positive_columns = [value for value in y_cols if value.__contains__('valor')]
df_ypred = pd.DataFrame(y_pred,columns=y_cols)
df_ypred[positive_columns] = df_ypred[positive_columns].abs()
df_modelo1 = pd.merge(df_X, df_ypred, left_index=True, right_index=True)
            """)

            st.write('#### Modelo 2')

            st.write('##### Definimos las columnas de fecha para X y el dolar blue para Y.')

            st.code("""
            X_cols = list(datos.columns[0:5])
y_cols = ['valor_dolar_blue']
X = datos[X_cols].values
y = datos[y_cols].values
X_train , X_test , y_train , y_test = train_test_split(X,y)
            """)

            st.write('##### Creamos un Modelo Ridge y lo testeamos.')

            st.code("""
            model = Ridge(random_state=50)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
MSE = mean_squared_error(y_test,y_pred)
RMSE = mean_squared_error(y_test,y_pred,squared= False)
R2 = r2_score(y_test,y_pred)
print('Error Cuadrado Medio: ', MSE.round(4))
print('Raiz del Error Cuadrado Medio: ', RMSE.round(4))
print('Coeficiente de Determinacion: ', R2.round(4))
            """)

            st.write("""
            Error Cuadrado Medio:  348.8301\n
Raiz del Error Cuadrado Medio:  18.677\n
Coeficiente de Determinacion:  0.8959
            """)

            st.write('##### Creamos un pipeline del modelo y exportamos.')

            st.code("""
            model = Ridge(random_state=50)
model = make_pipeline(StandardScaler(),model)
model.fit(X,y)
y_pred = model.predict(X)
dump(model, r'Modelos/Dolar Blue/Modelo 2/model.joblib')
            """)

            st.write('##### Creamos un DataFrame para el segundo modelo.')

            st.code("""
            df_X = pd.DataFrame(X,columns=X_cols)
positive_columns = [value for value in y_cols if value.__contains__('valor')]
df_ypred = pd.DataFrame(y_pred,columns=y_cols)
df_ypred[positive_columns] = df_ypred[positive_columns].abs()
df_modelo2 = pd.merge(df_X, df_ypred, left_index=True, right_index=True)
            """)

            st.write('#### Modelo 3')

            st.write('##### Extraemos todos los datos exceptuando la fecha del DataFrame del Modelo 1 y definimos el valor original del dolar del DataFrame original.')

            st.code("""
            X_cols = [value for value in list(df_modelo1.columns[5:])]
y_cols = ['valor_dolar_blue']
X = df_modelo1[X_cols].values
y = datos[y_cols].values
            """)

            st.write('##### Ajustamos los pesos para tomar las fechas mas altas y volatiles.')

            st.code("""
            sample_weight = 100 * np.abs(np.random.randn(X.shape[0]))
sample_weight = 100 * np.abs(np.random.randn(X.shape[0]))
n = 1
m = X.shape[0]
m10 = X.shape[0]/10
pesos_list= {
        'a' : [10000],
        'b' : [10000],
        'c' : [10000],
        'd' : [1000],
        'e' : [10000],
        'f' : [150000],
        'g' : [50000],
        'h' : [10000],
        'i' : [50000],
        'j' : [10000],
        'k' : [10000]
    }
pesos= {
        'a' : 0,
        'b' : 0,
        'c' : 0,
        'd' : 0,
        'e' : 0,
        'f' : 0,
        'g' : 0,
        'h' : 0,
        'i' : 0,
        'j' : 0,
        'k' : 0
}
pesos_list_new = True
_iters = 0
            """)

            st.write('##### Algoritmo de Ajuste Semi-Automatico de Pesos, uno define los valores esperados y el algoritmo aproxima los mejores pesos.')

            st.code("""
            _break = False
_debug = True
f_adj = 15
l_adj = 15
r2_adj = 0.1
rmse_adj = 5

for iter_out in range(1000):
    _iters+=1
    pesos_list = {key:[np.array(pesos_list[key]).mean().astype(int)] for key in pesos_list.keys()}
    if(_break):break
    if iter_out <= 950:
        f_adj *= 0.99
        l_adj *= 0.99
        r2_adj *= 0.99
        rmse_adj *= 0.99
    else:
        f_adj = 0
        l_adj = 0
        r2_adj = 0
        rmse_adj = 0



    for iter_in in range(100):
        _iters+=1
        pesos = {key:randint(
            (int(np.array(pesos_list[key]).mean().astype(int)-(np.array(pesos_list[key]).mean().astype(int)*0.15))),
            (int(np.array(pesos_list[key]).mean().astype(int)+(np.array(pesos_list[key]).mean().astype(int)*0.15)))
            ) for key in pesos_list.keys()}
        
        sample_weight[:int(m10)] = n*pesos['a']
        sample_weight[int(m10):int(m10*2)] = n*pesos['b']
        sample_weight[int(m10*2):int(m10*3)] = n*pesos['c']
        sample_weight[int(m10*3):int(m10*4)] = n*pesos['d']
        sample_weight[int(m10*4):int(m10*5)] = n*pesos['e']
        sample_weight[int(m10*5):int(m10*6)] = n*pesos['f']
        sample_weight[int(m10*6):int(m10*7)] = n*pesos['g']
        sample_weight[int(m10*7):int(m10*8)] = n*pesos['h']
        sample_weight[int(m10*8):int(m10*9)] = n*pesos['i']
        sample_weight[int(m10*9):int(m10*10)] = n*pesos['j']
        sample_weight[int(m-10):] = n*pesos['k']

        ###############################
        model = LinearRegression()
        model = make_pipeline(StandardScaler(),model)
        model.fit(X, y, **{'linearregression__sample_weight':sample_weight})
        y_pred = model.predict(X)
        #############################

        #############################
        df_X = pd.DataFrame(X,columns=X_cols)
        positive_columns = ['pred_oficial']
        df_ypred = pd.DataFrame(y_pred,columns=['pred_oficial'])
        df_ypred[positive_columns] = df_ypred[positive_columns].abs()
        df_oficial = pd.merge(df_X, df_ypred, left_index=True, right_index=True)
        ######################

        ######################
        comp = df_oficial.merge(datos.iloc[:,0:6],how='left')
        ##########################

        ##########################
        positive_columns = [value for value in y_cols if value.__contains__('valor')]
        df_ypred = pd.DataFrame(y_pred,columns=y_cols)
        df_ypred[positive_columns] = df_ypred[positive_columns].abs()
        df_modelo3 = pd.merge(datos[['Año','Semana','Mes','Dia_mes','Dia_semana']],df_ypred,right_index=True,left_index=True)
        df_modelo3_comp = df_modelo3.merge(datos[list(datos.columns)[0:5]+y_cols].rename(columns={'valor_dolar_blue':'dolar_blue_real'}),how='left')

        first = float(df_modelo3_comp['valor_dolar_blue'][:1])
        last = float(df_modelo3_comp['valor_dolar_blue'][-1:])

        first_real = float(df_modelo3_comp['dolar_blue_real'][:1])
        last_real = float(df_modelo3_comp['dolar_blue_real'][-1:])

        first_izq = first_real+2
        first_der = first_real+5

        last_izq = last_real+12
        last_der = last_real+20
        ##########################
            
        ##########################
        MSE = mean_squared_error(np.array(df_modelo3_comp['valor_dolar_blue']),np.array(df_modelo3_comp['dolar_blue_real']))
    
        RMSE = mean_squared_error(np.array(df_modelo3_comp['valor_dolar_blue']),np.array(df_modelo3_comp['dolar_blue_real']),squared= False)
        RMSE_izq = 1
        RMSE_der = 35
    
        R2 = r2_score(np.array(df_modelo3_comp['valor_dolar_blue']),np.array(df_modelo3_comp['dolar_blue_real']))
        R2_izq = 0.82
        R2_der = 0.92
    
        #######################

        #######################

        first_bool = (first_izq < first < first_der)
        last_bool = (last_izq < last < last_der)
        R2_bool = (R2_izq < round(R2,2) < R2_der)
        RMSE_bool = (RMSE_izq < round(RMSE,2) < RMSE_der)

        first_bool_str = '{}'.format((first_izq < first < first_der)).replace('True','✓').replace('False','⚠')
        last_bool_str = '{}'.format((last_izq < last < last_der)).replace('True','✓').replace('False','⚠')
        R2_bool_str = '{}'.format((R2_izq < round(R2,2) < R2_der)).replace('True','✓').replace('False','⚠')
        RMSE_bool_str = '{}'.format((RMSE_izq < round(RMSE,2) < RMSE_der)).replace('True','✓').replace('False','⚠')
        if ((first_izq-f_adj < first < first_der+f_adj) and (last_izq-l_adj < last < last_der+l_adj) and (R2_izq-r2_adj < round(R2,2) < R2_der+r2_adj) and (RMSE_izq < round(RMSE,2) < RMSE_der+rmse_adj)):
            if( (iter_out%4==0) and (iter_in==99)  and _debug):
                clear_output(wait=True) 
                print('=====================================================================================================================================================================')
                print('[first',first_bool_str,']:', round(first,2),'({})'.format(first_real),
                '[{} , {}]'.format(first_izq,first_der), 
                '\n[last ',last_bool_str,']:', round(last,2),'({})'.format(last_real),'[{} , {}]'.format(last_izq,last_der))
                print('[RMSE ',RMSE_bool_str,']: ', round(RMSE,2),'[{} , {}]'.format(RMSE_izq,RMSE_der),'\n[R2   ',R2_bool_str,']: ', round(R2,2),'[{} , {}]'.format(R2_izq,R2_der))
                print('[',first_bool_str,last_bool_str,RMSE_bool_str,R2_bool_str,']',' \nPesos:',pesos)
                print('=====================================================================================================================================================================')
            for key in pesos_list.keys():
                pesos_list[key].append(pesos[key])

        if(first_bool and last_bool and R2_bool and RMSE_bool):
            dump(model, r'Modelos/Dolar Blue/Modelo 3/model.joblib')
            print('==========================Balanceado====================')
            print('Error Cuadrado Medio: ', MSE.round(4))
            print('Raiz del Error Cuadrado Medio: ', RMSE.round(4))
            print('Coeficiente de Determinacion: ', R2.round(4))
            print('Iteraciones: ', _iters)
            _break = True

        if(_break):break


            """)
        
            st.write('#### Creamos el DataFrame final con las predicciones.')

            st.code("""
        predicciones = datos[datos.columns[0:5]].merge(df_modelo2,how='left').rename(columns={'valor_dolar_blue':'pred1_blue'})
predicciones = predicciones.merge(df_modelo3,how='left').rename(columns={'valor_dolar_blue':'pred2_blue'}).dropna()
predicciones['pred_diff'] = (predicciones['pred2_blue']-predicciones['pred1_blue'])
predicciones['pred_diff'] = predicciones['pred_diff'].abs()
predicciones = predicciones[list(predicciones.columns[0:6])+list(predicciones.columns[7:])+list(predicciones.columns[6:7])]
predicciones['pred_promedio'] = (predicciones['pred2_blue']+predicciones['pred1_blue'])/2
        """)

            st.write('#### Ejemplo del DataFrame resultante:')

            st.dataframe(pd.read_csv('Dolar_Blue/predicciones_2022_11_04.csv'))

            st.write('En la seccion grafico pueden verse los valores de manera interactiva.')

#########################################
# Blue Grafico
#########################################    

        if selected_blue == "Grafico":
            st.write('# Prediccion del Dolar Blue - 30 Dias')

            st.write('Fecha de Ejecucion: {}'.format(today.strftime('%Y/%m/%d')))

            st.altair_chart(
                (chart).interactive(),
                use_container_width=True
            )
    if selected_proyecto == 'Auto-ETL - Data Engineering':
        st.write('# En construccion')
        st.write("""
            ## [GitHub](https://github.com/CodeKova/Data_Engineering-Proyecto_Individual-Henry).
            """)
if selected == 'Contacto':

    components.html("""<script src="https://platform.linkedin.com/badges/js/profile.js" async defer type="text/javascript"></script>
    <div class="badge-base LI-profile-badge" data-locale="en_US" data-size="large" data-theme="dark" data-type="VERTICAL" data-vanity="thiago-ferster-924a0924a" data-version="v1" style="width:600px; margin:0 auto;"><a class="badge-base__link LI-simple-link" href="https://ar.linkedin.com/in/thiago-ferster-924a0924a?trk=profile-badge"></a></div>""",height=300,width=1000)

    st.markdown("<h1 style='font-size:200%;text-align: center; color: #353b4f;'>codekova@gmail.com </h1>", unsafe_allow_html=True)
