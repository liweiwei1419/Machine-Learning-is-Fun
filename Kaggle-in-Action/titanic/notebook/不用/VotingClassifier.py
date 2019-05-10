import os
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn import ensemble
from sklearn import model_selection

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

if __name__ == '__main__':



    '''
    要建立新的特征类目
    1. Fare Category
    2. Pclass Fare Category
    3. Family Size Category
    4. Age Group Category
    5. Name Length Category
    '''


    # 弃掉不需要的列
    def drop_col_not_req(df, cols):
        df.drop(cols, axis=1, inplace=True)


    # 建立Fare Category
    def fare_category(fare):
        if fare <= 4:
            return 'Very_Low_Fare'
        elif fare <= 10:
            return 'Low_Fare'
        elif fare <= 30:
            return 'Med_Fare'
        elif fare <= 45:
            return 'High_Fare'
        else:
            return 'Very_High_Fare'


    # 建立PClass Fare Category
    def pclass_fare_category(df, Pclass_1_mean_fare, Pclass_2_mean_fare, Pclass_3_mean_fare):
        if df['Pclass'] == 1:
            if df['Fare'] <= Pclass_1_mean_fare:
                return 'Pclass_1_Low_Fare'
            else:
                return 'Pclass_1_High_Fare'
        elif df['Pclass'] == 2:
            if df['Fare'] <= Pclass_2_mean_fare:
                return 'Pclass_2_Low_Fare'
            else:
                return 'Pclass_2_High_Fare'
        elif df['Pclass'] == 3:
            if df['Fare'] <= Pclass_3_mean_fare:
                return 'Pclass_3_Low_Fare'
            else:
                return 'Pclass_3_High_Fare'


    # 建立Family Size Category
    def family_size_category(family_size):
        if family_size <= 1:
            return 'Single'
        elif family_size <= 3:
            return 'Small_Family'
        else:
            return 'Large_Family'


    # 建立Age_Group_category
    def age_group_category(age):
        if age <= 1:
            return 'Baby'
        elif age <= 4:
            return 'Toddler'
        elif age <= 12:
            return 'Child'
        elif age <= 19:
            return 'Teenager'
        elif age <= 30:
            return 'Adult'
        elif age <= 50:
            return 'Middle_Aged'
        elif age < 60:
            return 'Senior_Citizen'
        else:
            return 'Old'


    # 建立Name_Length_Category
    def name_length_category(name_len):
        if name_len <= 19:
            return 'Very_Short_Name'
        elif name_len <= 28:
            return 'Short_Name'
        elif name_len <= 45:
            return 'Medium_Name'
        else:
            return 'Long_Name'


    # 填充NaN值
    # 使用GradientBoostingRegressor和LinearRegression来填充Age值
    def fill_missing_age(missing_age_train, missing_age_test):
        missing_age_X_train = missing_age_train.drop(['Age'], axis=1)
        missing_age_Y_train = missing_age_train['Age']
        missing_age_X_test = missing_age_test.drop(['Age'], axis=1)

        gbm_reg = ensemble.GradientBoostingRegressor(random_state=42)
        gbm_reg_param_grid = {'n_estimators': [2000], 'max_depth': [3],
                              'learning_rate': [0.01], 'max_features': [3]}
        gbm_reg_grid = model_selection.GridSearchCV(gbm_reg, gbm_reg_param_grid,
                                                    cv=10, n_jobs=25, verbose=1,
                                                    scoring='neg_mean_squared_error')
        gbm_reg_grid.fit(missing_age_X_train, missing_age_Y_train)
        print('Age feature Best GB Params:' + str(gbm_reg_grid.best_params_))
        print('Age feature Best GB Score:' + str(gbm_reg_grid.best_score_))
        print('GB Train Error for "Age" Feature Regressor:' + str(
            gbm_reg_grid.score(missing_age_X_train, missing_age_Y_train)))
        missing_age_test['Age_GB'] = gbm_reg_grid.predict(missing_age_X_test)
        print(missing_age_test['Age_GB'][:4])

        lrf_reg = LinearRegression()
        lrf_reg_param_grid = {'fit_intercept': [True], 'normalize': [True]}
        lrf_reg_grid = model_selection.GridSearchCV(lrf_reg, lrf_reg_param_grid,
                                                    cv=10, n_jobs=25, verbose=1,
                                                    scoring='neg_mean_squared_error')
        lrf_reg_grid.fit(missing_age_X_train, missing_age_Y_train)
        print('Age feature Best LR Params:' + str(lrf_reg_grid.best_params_))
        print('Age feature Best LR Score:' + str(lrf_reg_grid.best_score_))
        print('LR Train Error for "Age" Feature Regressor' + str(
            lrf_reg_grid.score(missing_age_X_train, missing_age_Y_train)))

        missing_age_test['Age_LRF'] = lrf_reg_grid.predict(missing_age_X_test)
        print(missing_age_test['Age_LRF'][:4])

        print('shape1', missing_age_test['Age'].shape, missing_age_test[['Age_GB', 'Age_LRF']].mode(axis=1).shape)
        # missing_age_test['Age'] = missing_age_test[['Age_GB','Age_LRF']].mode(axis=1)
        missing_age_test['Age'] = np.mean([missing_age_test['Age_GB'], missing_age_test['Age_LRF']])
        print(missing_age_test['Age'][:4])
        drop_col_not_req(missing_age_test, ['Age_GB', 'Age_LRF'])

        return missing_age_test


    # 筛选重要特征
    def get_top_n_features(titanic_train_data_X, titanic_train_data_Y, top_n_features):
        # 随机森林
        rf_est = RandomForestClassifier(random_state=42)
        rf_param_grid = {'n_estimators': [500], 'min_samples_split': [2, 3], 'max_depth': [20]}
        rf_grid = model_selection.GridSearchCV(rf_est, rf_param_grid, n_jobs=25, cv=10, verbose=1)
        rf_grid.fit(titanic_train_data_X, titanic_train_data_Y)

        print('Top N Features Best RF Params:' + str(rf_grid.best_params_))
        print('Top N Features Best RF Score:' + str(rf_grid.best_score_))
        print('Top N Features RF Train Error:' + str(rf_grid.score(titanic_train_data_X, titanic_train_data_Y)))

        feature_imp_sorted_rf = pd.DataFrame({'feature': list(titanic_train_data_X),
                                              'importance': rf_grid.best_estimator_.feature_importances_}).sort_values(
            'importance', ascending=False)
        features_top_n_rf = feature_imp_sorted_rf.head(top_n_features)['feature']
        print('Sample 25 Features from RF Classifier')
        print(str(features_top_n_rf[:25]))

        # AdaBoost
        ada_est = ensemble.AdaBoostClassifier(random_state=42)
        ada_param_grid = {'n_estimators': [500], 'learning_rate': [0.5, 0.6]}
        ada_grid = model_selection.GridSearchCV(ada_est, ada_param_grid, n_jobs=25, cv=10, verbose=1)
        ada_grid.fit(titanic_train_data_X, titanic_train_data_Y)

        print('Top N Features Best Ada Params:' + str(ada_grid.best_params_))
        print('Top N Features Best Ada Score:' + str(ada_grid.best_score_))
        print('Top N Features Ada Train Error:' + str(ada_grid.score(titanic_train_data_X, titanic_train_data_Y)))

        feature_imp_sorted_ada = pd.DataFrame({'feature': list(titanic_train_data_X),
                                               'importance': ada_grid.best_estimator_.feature_importances_}).sort_values(
            'importance', ascending=False)
        features_top_n_ada = feature_imp_sorted_ada.head(top_n_features)['feature']
        print('Sample 25 Feature from Ada Classifier:')
        print(str(features_top_n_ada[:25]))

        # ExtraTree
        et_est = ensemble.ExtraTreesClassifier(random_state=42)
        et_param_grid = {'n_estimators': [500], 'min_samples_split': [3, 4], 'max_depth': [15]}
        et_grid = model_selection.GridSearchCV(et_est, et_param_grid, n_jobs=25, cv=10, verbose=1)
        et_grid.fit(titanic_train_data_X, titanic_train_data_Y)

        print('Top N Features Best ET Params:' + str(et_grid.best_params_))
        print('Top N Features Best ET Score:' + str(et_grid.best_score_))
        print('Top N Features ET Train Error:' + str(et_grid.score(titanic_train_data_X, titanic_train_data_Y)))

        feature_imp_sorted_et = pd.DataFrame({'feature': list(titanic_train_data_X),
                                              'importance': et_grid.best_estimator_.feature_importances_}).sort_values(
            'importance', ascending=False)
        features_top_n_et = feature_imp_sorted_et.head(top_n_features)['feature']
        print('Sample 25 Features from ET Classifier:')
        print(str(features_top_n_et[:25]))

        # 融合以上三个模型
        features_top_n = pd.concat([features_top_n_rf, features_top_n_ada, features_top_n_et],
                                   ignore_index=True).drop_duplicates()

        return features_top_n


    # 开始训练模型

    train_data_org = pd.read_csv('./input/train.csv')
    test_data_org = pd.read_csv('./input/test.csv')

    # 将训练集和测试集合并

    test_data_org['Survived'] = 0
    combined_train_test = train_data_org.append(test_data_org)

    # 特征工程

    # 1. Embarked
    # Embarkde中有两个NaN
    if combined_train_test['Embarked'].isnull().sum() != 0:
        combined_train_test['Embarked'].fillna(combined_train_test['Embarked'].mode().iloc[0], inplace=True)

    emb_dummies_df = pd.get_dummies(combined_train_test['Embarked'],
                                    prefix=combined_train_test[['Embarked']].columns[0])
    combined_train_test = pd.concat([combined_train_test, emb_dummies_df], axis=1)

    # 2. Sex
    sex_dummies_df = pd.get_dummies(combined_train_test['Sex'], prefix=combined_train_test[['Sex']].columns[0])
    combined_train_test = pd.concat([combined_train_test, sex_dummies_df], axis=1)

    # 3. Name
    # Title
    combined_train_test['Title'] = combined_train_test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    # print(combined_train_test['Title'].groupby(by=combined_train_test['Title']).count().sort_values(ascending=False))

    title_Dict = {}
    title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
    title_Dict.update(dict.fromkeys(['Jonkheer', 'Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
    title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
    title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
    title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
    title_Dict.update(dict.fromkeys(['Master'], 'Master'))

    combined_train_test['Title'] = combined_train_test['Title'].map(title_Dict)
    # print(combined_train_test['Title'].groupby(by=combined_train_test['Title']).count().sort_values(ascending=False))

    title_dummies_df = pd.get_dummies(combined_train_test['Title'], prefix=combined_train_test[['Title']].columns[0])
    combined_train_test = pd.concat([combined_train_test, title_dummies_df], axis=1)
    # print(combined_train_test)

    # Name Length
    combined_train_test['Name_Length'] = combined_train_test['Name'].str.len()
    combined_train_test['Name_length_Category'] = combined_train_test['Name_Length'].map(name_length_category)
    # print(combined_train_test['Name_length_Category'].groupby(by=combined_train_test['Name_length_Category']).count().sort_values(ascending=False))

    le_name = LabelEncoder()
    le_name.fit(np.array(['Very_Short_Name', 'Short_Name', 'Medium_Name', 'Long_Name']))
    combined_train_test['Name_length_Category'] = le_name.transform(combined_train_test['Name_length_Category'])

    # print(combined_train_test[['Name_length_Category','Survived']].corr())
    name_length_dummies_df = pd.get_dummies(combined_train_test['Name_length_Category'],
                                            prefix=combined_train_test[['Name_length_Category']].columns[0])
    combined_train_test = pd.concat([combined_train_test, name_length_dummies_df], axis=1)
    # print(combined_train_test)

    # First Name
    combined_train_test['First_Name'] = combined_train_test['Name'].str.extract('^(.+?),')
    first_name_dummies_df = pd.get_dummies(combined_train_test['First_Name'],
                                           prefix=combined_train_test[['First_Name']].columns[0])
    combined_train_test = pd.concat([combined_train_test, first_name_dummies_df], axis=1)
    # print(combined_train_test)

    # Last_Name
    combined_train_test['Last_Name'] = combined_train_test['Name'].str.split('\.')
    combined_train_test['Last_Name'] = combined_train_test['Last_Name'].str.strip('\([^)]*\)')
    combined_train_test['Last_Name'].fillna(combined_train_test['Name'].str.split('\.').str[1])
    # print(combined_train_test['Last_Name'].groupby(by = combined_train_test['Last_Name']).count().sort_values(ascending = False)[:5])
    last_name_dummies_df = pd.get_dummies(combined_train_test['Last_Name'],
                                          prefix=combined_train_test[['Last_Name']].columns[0])
    combined_train_test = pd.concat([combined_train_test, last_name_dummies_df], axis=1)

    # print(combined_train_test)

    # Original_Name
    combined_train_test['Original_Name'] = combined_train_test['Name'].str.split('\((.*?)\)').str[1].str.strip(
        '\"').str.strip()
    # print(combined_train_test['Original_Name'].groupby(by = combined_train_test['Original_Name']).count().sort_values(ascending = False)[:5])
    original_name_dummies_df = pd.get_dummies(combined_train_test['Original_Name'],
                                              prefix=combined_train_test[['Original_Name']].columns[0])
    combined_train_test = pd.concat([combined_train_test, original_name_dummies_df], axis=1)

    # 4. Fare
    # 填充NaN
    if combined_train_test['Fare'].isnull().sum() != 0:
        combined_train_test['Fare'] = combined_train_test[['Fare']].fillna(
            combined_train_test.groupby('Pclass').transform('mean'))

    # 将多人船票的价格平均到每人
    combined_train_test['Group_Ticket'] = combined_train_test['Fare'].groupby(
        by=combined_train_test['Ticket']).transform('count')
    combined_train_test['Fare'] = combined_train_test['Fare'] / combined_train_test['Group_Ticket']
    combined_train_test.drop(['Group_Ticket'], axis=1, inplace=True)

    # 去除Fare为0的项
    if sum(n == 0 for n in combined_train_test.Fare.values.flatten()) > 0:
        combined_train_test.loc[combined_train_test.Fare == 0, 'Fare'] = np.nan
        combined_train_test['Fare'] = combined_train_test[['Fare']].fillna(
            combined_train_test.groupby('Pclass').transform('mean'))
    # print(combined_train_test['Fare'].describe())
    # 建立Fare Category
    combined_train_test['Fare_Category'] = combined_train_test['Fare'].map(fare_category)
    le_fare = LabelEncoder()
    le_fare.fit(np.array(['Very_Low_Fare', 'Low_Fare', 'Med_Fare', 'High_Fare', 'Very_High_Fare']))
    combined_train_test['Fare_Category'] = le_fare.transform(combined_train_test['Fare_Category'])

    fare_cat_dummies_df = pd.get_dummies(combined_train_test['Fare_Category'],
                                         prefix=combined_train_test[['Fare_Category']].columns[0])
    combined_train_test = pd.concat([combined_train_test, fare_cat_dummies_df], axis=1)
    # print(combined_train_test['Fare_Category'].groupby(by = combined_train_test['Fare_Category']).count().sort_values(ascending = False))

    # 5. Pclass
    # print(combined_train_test['Fare'].groupby(by = combined_train_test['Pclass']).mean())
    Pclass_1_mean_fare = combined_train_test['Fare'].groupby(by=combined_train_test['Pclass']).mean().get([1]).values[0]
    Pclass_2_mean_fare = combined_train_test['Fare'].groupby(by=combined_train_test['Pclass']).mean().get([2]).values[0]
    Pclass_3_mean_fare = combined_train_test['Fare'].groupby(by=combined_train_test['Pclass']).mean().get([3]).values[0]
    # 建立Pclass_Fare Category
    combined_train_test['Pclass_Fare_Category'] = combined_train_test.apply(pclass_fare_category, args=(
        Pclass_1_mean_fare, Pclass_2_mean_fare, Pclass_3_mean_fare), axis=1)
    # print(combined_train_test['Pclass_Fare_Category'].groupby(by = combined_train_test['Pclass_Fare_Category']).count().sort_values(ascending = False))
    p_fare = LabelEncoder()
    p_fare.fit(np.array(
        ['Pclass_1_Low_Fare', 'Pclass_1_High_Fare', 'Pclass_2_Low_Fare', 'Pclass_2_High_Fare', 'Pclass_3_Low_Fare',
         'Pclass_3_High_Fare']))
    combined_train_test['Pclass_Fare_Category'] = p_fare.transform(combined_train_test['Pclass_Fare_Category'])

    # 6. Parch and SibSp

    combined_train_test['Family_Size'] = combined_train_test['Parch'] + combined_train_test['SibSp'] + 1
    combined_train_test['Family_Size_Category'] = combined_train_test['Family_Size'].map(family_size_category)

    le_family = LabelEncoder()
    le_family.fit(np.array(['Single', 'Small_Family', 'Large_Family']))
    combined_train_test['Family_Size_Category'] = le_family.transform(combined_train_test['Family_Size_Category'])

    fam_size_cat_dummies_df = pd.get_dummies(combined_train_test['Family_Size_Category'],
                                             prefix=combined_train_test[['Family_Size_Category']].columns[0])
    combined_train_test = pd.concat([combined_train_test, fam_size_cat_dummies_df], axis=1)
    # print(combined_train_test)

    # 7. Age
    # 填充Age中的NaN值

    combined_train_test['Age_Null'] = combined_train_test['Age'].apply(lambda x: 1 if pd.notnull(x) else 0)

    missing_age_df = pd.DataFrame(
        combined_train_test[['Age', 'Parch', 'Sex', 'SibSp', 'Family_Size', 'Family_Size_Category',
                             'Title', 'Fare', 'Fare_Category', 'Pclass', 'Embarked']])
    missing_age_df = pd.get_dummies(missing_age_df,
                                    columns=['Title', 'Family_Size_Category', 'Fare_Category', 'Sex', 'Pclass',
                                             'Embarked'])
    missing_age_train = missing_age_df[missing_age_df['Age'].notnull()]
    missing_age_test = missing_age_df[missing_age_df['Age'].isnull()]
    combined_train_test.loc[(combined_train_test.Age.isnull()), 'Age'] = fill_missing_age(missing_age_train,
                                                                                          missing_age_test)
    # print(combined_train_test.describe())

    # 检查是否有异常值
    if sum(n < 0 for n in combined_train_test.Age.values.flatten()) > 0:
        combined_train_test.loc[combined_train_test.Age < 0, 'Age'] = np.nan
        combined_train_test['Age'] = combined_train_test[['Age']].fillna(
            combined_train_test.groupby('Title').transform('mean'))
    # print(combined_train_test['Age'].groupby(by=combined_train_test['Title']).mean().sort_values(ascending=True))

    # 建立Age_Category
    combined_train_test['Age_Category'] = combined_train_test['Age'].map(age_group_category)
    le_age = LabelEncoder()
    le_age.fit(np.array(['Baby', 'Toddler', 'Child', 'Teenager', 'Adult', 'Middle_Aged', 'Senior_Citizen', 'Old']))
    combined_train_test['Age_Category'] = le_age.transform(combined_train_test['Age_Category'])
    age_cat_dummies_df = pd.get_dummies(combined_train_test['Age_Category'],
                                        prefix=combined_train_test[['Age_Category']].columns[0])
    combined_train_test = pd.concat([combined_train_test, age_cat_dummies_df], axis=1)

    # 8. Ticket
    combined_train_test['Ticket_Letter'] = combined_train_test['Ticket'].str.split().str[0]
    combined_train_test['Ticket_Letter'] = combined_train_test['Ticket_Letter'].apply(
        lambda x: np.nan if x.isnumeric() else x)
    combined_train_test['Ticket_Number'] = combined_train_test['Ticket'].apply(
        lambda x: pd.to_numeric(x, errors='coerce'))
    combined_train_test['Ticket_Number'].fillna(0, inplace=True)
    combined_train_test = pd.get_dummies(combined_train_test, columns=['Ticket', 'Ticket_Letter'])
    # print(combined_train_test.shape)

    # 9. Cabin
    combined_train_test['Cabin_Letter'] = combined_train_test['Cabin'].apply(
        lambda x: str(x)[0] if pd.notnull(x) else x)
    combined_train_test = pd.get_dummies(combined_train_test, columns=['Cabin', 'Cabin_Letter'])
    # print(combined_train_test.shape)

    # 10. 将Age和Fare正则化
    scale_age_fare = preprocessing.StandardScaler().fit(combined_train_test[['Age', 'Fare']])
    combined_train_test[['Age', 'Fare']] = scale_age_fare.transform(combined_train_test[['Age', 'Fare']])

    # 11. 弃掉无用列
    combined_train_test.drop(['Name', 'PassengerId', 'Embarked', 'Sex', 'Title', 'Fare_Category',
                              'Family_Size_Category', 'Age_Category', 'First_Name', 'Last_Name',
                              'Original_Name', 'Name_length_Category'], axis=1, inplace=True)
    print(combined_train_test.describe())
    # 12. 整理数据

    train_data = combined_train_test[:891]
    test_data = combined_train_test[891:]

    titanic_train_data_X = train_data.drop(['Survived'], axis=1)
    titanic_train_data_Y = train_data['Survived']

    titanic_test_data_X = test_data.drop(['Survived'], axis=1)

    # 13. 利用特征值重要性排名来去除无用列
    feature_to_pick = 250
    feature_top_n = get_top_n_features(titanic_train_data_X, titanic_train_data_Y, feature_to_pick)
    print('Total Feature:' + str(combined_train_test.shape))
    print('Picked Feature' + str(feature_top_n.shape))

    titanic_train_data_X = titanic_train_data_X[feature_top_n]
    del titanic_train_data_X['Ticket_Number']
    titanic_test_data_X = titanic_test_data_X[feature_top_n]
    del titanic_test_data_X['Ticket_Number']

    # 14.建立模型
    rf_est = ensemble.RandomForestClassifier(n_estimators=750, criterion='gini', max_features='sqrt',
                                             max_depth=3, min_samples_split=4, min_samples_leaf=2,
                                             n_jobs=50, random_state=42, verbose=1)
    gbm_est = ensemble.GradientBoostingClassifier(n_estimators=900, learning_rate=0.0008, loss='exponential',
                                                  min_samples_split=3, min_samples_leaf=2, max_features='sqrt',
                                                  max_depth=3, random_state=42, verbose=1)
    et_est = ensemble.ExtraTreesClassifier(n_estimators=750, max_features='sqrt', max_depth=35, n_jobs=50,
                                           criterion='entropy', random_state=42, verbose=1)

    voting_est = ensemble.VotingClassifier(estimators=[('rf', rf_est), ('gbm', gbm_est), ('et', et_est)],
                                           voting='soft', weights=[3, 5, 2],
                                           n_jobs=50)
    voting_est.fit(titanic_train_data_X, titanic_train_data_Y)
    print('VotingClassifier Score:' + str(voting_est.score(titanic_train_data_X, titanic_train_data_Y)))
    print('VotingClassifier Estimators:' + str(voting_est.estimators_))

    # 预测
    titanic_test_data_X['Survived'] = voting_est.predict(titanic_test_data_X)

    submission = pd.DataFrame({'PassengerId': test_data_org.loc[:, 'PassengerId'],
                               'Survived': titanic_test_data_X.loc[:, 'Survived']})

    submission.to_csv('submission_result.csv', index=False, sep=',')
