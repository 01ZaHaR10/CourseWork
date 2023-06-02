from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling

# модель Байесовской сети
model = BayesianModel()

# Создание узлов
model.add_nodes_from(['age', 'gender', 'smoking', 'disease', 'air_pollution'])

# Задаем ребра
model.add_edges_from([
    ('age', 'smoking'),
    ('gender', 'smoking'),
    ('smoking', 'disease'),
    ('air_pollution', 'disease')
])

# Задаем условные вероятности

cpd_age = TabularCPD(variable='age', variable_card=3,
                     values=[[0.21], [0.63], [0.16]],
                     state_names={'age': ['молодой', 'средний', 'пожилой']})

cpd_gender = TabularCPD(variable='gender', variable_card=2,
                        values=[[0.46], [0.54]],
                        state_names={'gender': ['мужской', 'женский']})

cpd_smoking = TabularCPD(variable='smoking', variable_card=2,
                         values=[[0.0979, 0.0438, 0.2967, 0.1326, 0.0754, 0.0337],
                                 [0.9021, 0.9562, 0.7033, 0.8674, 0.9246, 0.9663]],
                         evidence=['age', 'gender'],
                         evidence_card=[3, 2],
                         state_names={
                             'smoking': ['да', 'нет'],
                             'age': ['молодой', 'средний', 'пожилой'],
                             'gender': ['мужской', 'женский']
                         })

cpd_disease = TabularCPD(variable='disease', variable_card=2,
                         values=[[0.9, 0.7, 0.4, 0.6, 0.3, 0.1],
                                 [0.1, 0.3, 0.6, 0.4, 0.7, 0.9]],
                         evidence=['smoking', 'air_pollution'],
                         evidence_card=[2, 3],
                         state_names={'disease': ['присутствует', 'отсутствует'],
                                      'smoking': ['да', 'нет'],
                                      'air_pollution': ['высокое', 'среднее', 'низкое']})

cpd_air_pollution = TabularCPD(variable='air_pollution', variable_card=3,
                              values=[[0.2], [0.3], [0.5]],
                              state_names={'air_pollution': ['высокое', 'среднее', 'низкое']})


model.add_cpds(cpd_age, cpd_gender, cpd_smoking, cpd_air_pollution, cpd_disease)

model.check_model()

# Создаем объект для генерации данных на основе модели
sampler = BayesianModelSampling(model)

data = sampler.forward_sample(size=1000)

data_list = list(data.reset_index(drop=True).values)

with open("generated_data.txt", "w") as file:
    for item in data_list:
        file.write(','.join(map(str, item)) + '\n')