import numpy as np
import pandas as pd
import pandasql
import json
import requests


#Setup Virtual Environment (Da CMD):
> python -m venv nome_env
#Per attivare il VENV:
#> nome_env\Scripts\activate.bat

#Installare un modulo:
python -m pip install numpy
#Oppure:
python -m easy_install numpy
python -m easy_install pandas

#Dataset
#dictionary items count
len(yourdict)

#NUMPY

# ARRAY
array = np.array([1, 4, 5, 8], float)
array = np.array([[1, 2, 3], [4, 5, 6]], float)  # a 2D array/Matrix
print array[:2]
array[1] = 5.0

two_D_array = np.array([[1, 2, 3], [4, 5, 6]], float)
print two_D_array[1, :]



# Array operations
array_1 = np.array([1, 2, 3], float)
array_2 = np.array([5, 2, 6], float)
print array_1 + array_2		# Somma i valori uno ad uno
print array_1 - array_2
print array_1 * array_2		# Moltiplica i valori 1 a 1

print np.mean(array_1)		# Media
print np.dot(array_1, array_2)	#Moltiplicazione riga x colonna



#PANDAS

#Series
#---------------------------------------------------
series = pd.Series(['Dave', 'Cheng-Han', 359, 9001],
                   index=['Instructor', 'Curriculum Manager',
                          'Course Number', 'Power Level'])
print series
print series['Instructor']												# Ritorna un valore (o serie di valori)
print series[['Instructor', 'Curriculum Manager', 'Course Number']]		# Ritorna una serie

print len(series) # conta gli elementi

# boolean selectors
cuteness = pd.Series([1, 2, 3, 4, 5], index=['Cockroach', 'Fish', 'Mini Pig',
                                             'Puppy', 'Kitten'])
print cuteness > 3 					# Ritorna una serie con gloi stessi indici, ma con un booleano al posto del numero
#Cockroach    False
#Fish         False
#Mini Pig     False
#Puppy         True
#Kitten        True
#dtype: bool

print cuteness[cuteness > 3]		#Filtra la serie
#Puppy     4
#Kitten    5
#dtype: int64



#DataFrame = Serie di Serie
#---------------------------------------------------
countries = ['Russian Fed.', 'Norway', 'Canada', 'United States',
                 'Netherlands', 'Germany', 'Switzerland', 'Belarus',
                 'Austria', 'France', 'Poland', 'China', 'Korea', 
                 'Sweden', 'Czech Republic', 'Slovenia', 'Japan',
                 'Finland', 'Great Britain', 'Ukraine', 'Slovakia',
                 'Italy', 'Latvia', 'Australia', 'Croatia', 'Kazakhstan']

gold = [13, 11, 10, 9, 8, 8, 6, 5, 4, 4, 4, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
silver = [11, 5, 10, 7, 7, 6, 3, 0, 8, 4, 1, 4, 3, 7, 4, 2, 4, 3, 1, 0, 0, 2, 2, 2, 1, 0]
bronze = [9, 10, 5, 12, 9, 5, 2, 1, 5, 7, 1, 2, 2, 6, 2, 4, 3, 1, 2, 1, 0, 6, 2, 1, 0, 1]

df = pd.DataFrame({'country_name' : countries),
                 'gold' : pd.Series(gold),
                 'silver' : pd.Series(silver),
                 'bronze' : pd.Series(bronze)}) 

# Filtri
print df['gold']		#Ritorna una serie
print df.iloc[[0]]				# Il primo record (SEMPRE con doppie quadre)
print df.loc[[0]]				# come sopra
print df[3:5]					# Intervallo di record
print df[df.gold > 0]			# Record filtrati per medaglia d'oro
print df[(df.gold > 0) & (df.silver == 1)]


# Media
print df[['gold', 'silver', 'bronze']].apply(np.mean)
# gold      3.807692
# silver    3.730769
# bronze    3.807692
# dtype: float64

print np.mean(df["gold"])	# Ritorna un float


#giorno della settimana
df['weekday'] = df['my_dt'].apply(lambda x: x.weekday())


points = np.dot(df[['gold', 'silver', 'bronze']], [4, 2, 1])		# Moltiplicazione di matrici
print points
points = map(lambda g,s,b : g*4 + s*2 + b, gold, silver, bronze)	# Equivalente a quella sopra, in questo caso
print points

print pd.DataFrame({'country_name': pd.Series(countries),
                'points': pd.Series(points) })




# Files
file_path = 'D:\\Erni-OneDrive\\OneDrive\\Nanodegree\\Python\\titanic_data.csv'
# Read
df = pd.read_csv(file_path)
# Write
df.to_csv(file_path)

predictions = {}

for passenger_index, passenger in df.iterrows():
    passenger_id = passenger['PassengerId']

    if passenger['Sex'] == 'male':
        predictions[passenger_id] = 0
    else:
        predictions[passenger_id] = 1


# Concatenazione di valori in un DataFrame
df['nameFull'] = (df['nameFirst'] + ' ' + df['nameLast'])

# Modifica dei nomi delle colonne
df.rename(columns = lambda x: x.replace(' ', '_').lower(), inplace=True)

# SQL in data frames:
q = """
-- YOUR QUERY HERE
SELECT * FROM ds
"""

# CAST in SQL to get integers:
q = """... where cast(maxtempi as integer) = 76"""

#Execute your SQL command against the pandas frame
sql_result = pandasql.sqldf(q.lower(), locals())

q = """
    SELECT COUNT(*)
    FROM weather_data
    WHERE cast(rain as integer) = 1
"""
# get day of week: 'strftime' keyword in SQL.
#    For example, 
q = "cast (strftime('%w', date) as integer)" 
# will return 0 if the date is a Sunday or 6 if the date is a Saturday.


#Mean function in SQL: AVG()
q = """
    SELECT AVG(cast(meantempi as integer))
    FROM weather_data
    WHERE cast (strftime('%w', date) as integer) = 6 OR cast (strftime('%w', date) as integer) = 0
    """
    
q = """
    SELECT AVG(cast (mintempi AS integer))
    FROM weather_data
    WHERE cast(rain AS integer) = 1 AND mintempi > 55
    """


# API
url = "http://www...."
data = requests.get(url)
data = json.loads(data.text)

data2 = requests.post(url, data={"key": "value"})

#Using session for the requests:
s = requests.Session()
r = s.get(url)
r = s.post(url, data)

top_artist = data['topartists']['artist'][0]['name']
# Per stampare i dati ben formattati, usare pprint -> https://docs.python.org/2/library/pprint.html
print data #tipo Dictionary
print data['artist']




# Data wrangling
df.describe()       #Descrive i dati di un dataframe

# Impute values: Replace missing values (na) with
df['column'] = df['column'].fillna(12)
baseball['weight'] = baseball['weight'].fillna(numpy.mean(baseball['weight']))


# new column with shifted values from another column
df['ENTRIESn_previeus'] = df['ENTRIESn'].shift(1)


# new column with shifted values from another column, grouped by some columns
newSeries = df.groupby(by=['SCP', 'C/A', 'UNIT'])['ENTRIESn'].shift(1)
#il gropuby(...) ritorna una Serie, con indice una Tupla
#Per avere l'indice del gropuby in una tupla, bisogna fare
pandas.Series(newSeries.index.values)
#Oppure se è ragruppato per più coilonne, bisogna prenderle una ad una dalla tupla
pandas.Series([x[0] for x in newSeries.index.values])


#drop column
df = df.drop('ENTRIESn_previeus', 1)

# CSV 
name = "D:\Erni-OneDrive\OneDrive\Nanodegree\Python\Data\\turnstile_110528.txt"

with open(name) as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    with open('updated_' + name, 'w+') as outputfile:
        csvwriter = csv.writer(outputfile, delimiter=',')
        
        for row in csvreader:
                        
            for i in range(3, len(row), 5):
                r = [row[0], row[1], row[2], row[i], row[i+1], row[i+2], row[i+3], row[i+4].rstrip()]
                csvwriter.writerow(r)
                #print r


# File read and write without csv
with open(output_file, 'w') as master_file:
   master_file.write('C/A,UNIT,SCP,DATEn,TIMEn,DESCn,ENTRIESn,EXITSn\n')
   for filename in filenames:
        with open(filename) as csvfile:
            for line in csvfile:
                master_file.write(line)


with open(datafile, "r") as f:
    header = f.readline().split(',')
    for i, line in enumerate(f):
        line_data = line.split(',')


#problema empty line - se il csv stampa una riga vuota, usare lineterminator='\n':

    with open(output_bad, "w") as g:
        writer = csv.DictWriter(g, delimiter=",", fieldnames= header, lineterminator='\n')
        writer.writeheader()
        for row in bad:
            writer.writerow(row)

#Datetime
#import datetime
date = datetime.strptime("11/03/1988", "format")

d = datetime.datetime.strptime(date, '%m-%d-%y')

date_formatted = d.strftime('%Y-%m-%d')



#Statistica
import scipy.stats 

scipy.stats.ttest_ind(list_1, list_2, equal_var=False)
# Ritorna una tupla con 2 valori: 
# t = t-value
# p = il valore di p per un test t a 2 code

df = pandas.read_csv(filename)

r = df[df.handedness == 'R']['avg']
l = df[df.handedness == 'L']['avg']
p = 0.05

#print r.describe()
#print l.describe()

t_test = scipy.stats.ttest_ind(r, l, equal_var=False)
the_samples_are_equal = (t_test[1] > p) 
return (the_samples_are_equal, t_test)


# mann-whitney u test 
u_value, p_value = scipy.stats.mannwhitneyu(l, r)


# shapiro-wilk w test (per sapere se il campione ha una distribuzione normale)
w, p = scipy.stats.shapiro(data)



# PLOT
import matplotlib.pyplot as plt

plt.figure()
plt.xlabel('ENTRIESn_hourly')
plt.ylabel('Frequency')
turnstile_weather[turnstile_weather.rain == 0]['ENTRIESn_hourly'].hist(bins=50, range=(0,2000), color='r', label='No rain') # your code here to plot a historgram for hourly entries when it is raining
turnstile_weather[turnstile_weather.rain == 1]['ENTRIESn_hourly'].hist(bins=50, range=(0,2000), color='b', label='Rain') # your code here to plot a historgram for hourly entries when it is not raining
plt.legend()

# Sample plot of an array
plt.figure()

arr = pandas.Series(prediction_error)    
arr.hist(bins = 50, range = (-7000, 7000))

plt.show()

#GGPLOT
gg = ggplot(aes(x='yearID', y='HR'), data=df) + geom_point(color='red') + geom_line(color = 'red') + ggtitle('title') + xlab('label x') + ylab('label y') + \
    scale_x_discrete(breaks=(0,1,2,3,4,5,6), labels=("Sun","Mon","Twe","Wed","Thu","Fri","Sat"))


# color code records with different value in teamID:
gg = ggplot(aes(x='yearID', y='HR', color='teamID'), data = df) + geom_point() + geom_line()
    



# DUMMY -> Crea una colonna per ogni valore univoco nella colonna 'UNIT' e riempie la matrice con 0 e 1 
dummy_units = pandas.get_dummies(dataframe['UNIT'], prefix='unit')

# Join DF
features = features.join(dummy_units)


# Strip non ascii characters:
pattern = re.compile('\W')
stripped_word = re.sub(pattern, '', word) #replace
# Oppure
key = i.translate(string.maketrans("",""),string.punctuation).lower()


# Format string
print "{0}\t{1}".format('value 1', 'value 2')  # Formatta con un tab in mezzo alle due stringhe


#Return
break

#Parse HTML
from bs4 import BeautifulSoup

soup = BeautifulSoup(open("filename.html"))
someList = soup.find(id = "someElementId")
for item in someList.find_all("option"):
    my_array.append(item['value'])
    
    
#Check if a string contains a number
a = "03523"
a.isdigit()
int(a)


# Mongo DB
    from pymongo import MongoClient
    client = MongoClient('localhost:27017')
    # 'examples' here is the database name. It will be created if it does not exist.
    db = client.examples
    db.cities.insert({"name" : "Chicago"})
    query = { "name": "Trieste" }
    query = { "population": { "$gt": 500000, "$lt": 100000000 } }
    query = { "foundingDate": { "$gte": datetime(1990, 1, 1) } }
    query = { "foundingDate": { "$exists": 1 } }
    query = { "foundingDate": { "$regex": "[Ff]riendship|[Hh]appiness" } }
    query = { "model_year" : { "$in": [1965, 1966, 1967] } }
    query = { "model_year" : { "$all": [1965, 1966, 1967] } }
    query = { "manufacturer": "Ford Motor Company", "assembly": { "$in": ["Germany", "United Kingdom", "Japan"]} }
    query = { "dimensions.weight": { "$gt": 500 } }
    
    
    p = db.cities.find(query)
    p = db.cities.find(query).count()
    
    #find one and save
    city = db.cities.find_one(query)
    city["coutryCode"] = "DEU"
    db.cities.save(city)

    #UPDATE
    db.cities.update(query, { "$set": { "countryCode": "DEU" } })    
    db.cities.update(query, { "$unset": { "countryCode": "" } }) #se il countryCode c'è, viene cancellato  
    db.cities.update(query, { "$set": { "countryCode": "DEU" } }, multi= True)    
    
    db.cities.update(query, newDocument)    
    
    #Remove all:
    db.cities.drop() #Drops everything efficiently
    db.cities.remove(query)
    
#Import data into MOngodb
mongoimport -db dbname -c collectionname --file input-file.json

#MongoDB aggregatoin
    db.tweets.aggregate([
        { "$group": { "_id": "$user.schreen_name",
                     "count" : { "$sum": 1 } } }, #Operators: $sum, $max, $min, $first, $last, $avg
                     #Array operaqtors: $push, $addToSet (come push, ma non inserisce duplicati)
        { "$sort": { "count": -1 } }, 
        { "$project": { "ratio": { "$divide": ["$user.followers_count", "$user.friends_count"]}, 
                       "screen_name": "$user.screen_name"} },
        { "$match": { "user.friends_count": { "$gt": 0 } }, "some_array": { "$size" : 3 } },
        { "$skip": 1},
        { "$limit": 1},
        { "$unwind": "$entities.user_mentions"} #Se un oggetto ha un field con un array di dati, 
                        # unwind lo spezza in duplicati dello stesso oggetto, 
                        # ogniuno con uno dei valori del array
        ])
    
    my_collection.create_index([("mike", pymongo.DESCENDING),
                                ("eliot", pymongo.ASCENDING)])
    db.tweets.ensure_index({"name": pymongo.ASCENDING}) # crea un indice
    
    #Per poter avere un indice geospaziale è mecessario salvar i dati nel formato
    # location: [x, y] 
    db.tweets.ensure_index([("loc", pymongo.GEO2D)]) # crea un indice
    
    #Un indice geospaziale abilita l'operatore: $near
    db.nodes.find( { "loc": { "$near": [42.1, 34.2]}})
       
# xml read

    tree = ET.parse(filename)
    root = tree.getroot()
    
    for node in tree.iter():
        print node.tag

# oppure parse xml al volo
    for event, element in ET.iterparse(filename, events=("start",)):
        print element.tag
        for tag in element.iter("nodename"):
            pass

# For - Next:
for i in items:
    if i%10:
        continue
    if i%1000:
        break


################################
# Machine learning
################################    

# Get a subset of observations   
features_train = features_train[:len(features_train)/100] 


### Naive Bayes algorithm

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

#Fit model
clf.fit(features_train, labels_train)

#Predict
prediction = clf.predict(features_test)

#Score
from sklearn.metrics import accuracy_score
print accuracy_score(labels_test, prediction)



### SVM - Support Vector Machine

from sklearn.svm import SVC
clf = SVC(kernel = 'linear')



### Decision tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(min_samples_split=50)


clf.fit(features_train, labels_train)

prediction = clf.predict(features_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(labels_test, prediction)
### be sure to compute the accuracy on the test set

### Regression

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(ages_train, net_worths_train)
reg.predict([1,2])[0]
reg.intercept_[0]
reg.coef_[0][0]
reg.score(input_test, expected) #R square score


### Lasso regression with automatic feature selection
>>> from sklearn import linear_model
>>> clf = linear_model.Lasso(alpha=0.1)
>>> clf.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,
   normalize=False, positive=False, precompute=False, random_state=None,
   selection='cyclic', tol=0.0001, warm_start=False)
>>> print(clf.coef_)
[ 0.85  0.  ]
>>> print(clf.intercept_)
0.15

### Unsupervised learning - clustering
from sklearn.cluster import KMeans
features_list = ["poi", feature_1, feature_2]
data2 = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data2 )
clf = KMeans(n_clusters=2)
pred = clf.fit_predict( finance_features )