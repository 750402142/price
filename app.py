
import joblib
import streamlit as st
import pandas as pd
import numpy as np
data0 = pd.read_csv('链家二手房房源.csv',index_col=0)#原始数据
data0 =data0[data0['朝向']!='5室0厅2卫'].reset_index(drop=True)
data = pd.read_csv('项目二二手房数据.csv',index_col=0)#处理后进行预测的样本数据
st.set_page_config(layout="wide")
st.title('🤖 Machine Learning App',)
st.markdown("""
    <style>
    h1 {
        text-align: center;
    }
    .header{
        background-color: #ADD8E6;
    }
    </style>
""", unsafe_allow_html=True)
# st.info('这是一个对基于房价数据,用线性回归,岭回归,拉索回归,决策数与随机森林的回归与分类,\n'
#         'AdaBoostRegressor,DecisionTreeRegressor,GradientBoostingRegressor,\n'
#         '这些模型进行数据评估,最后分别得到每个模型的最佳模型,并可以对输入特征预测的app')
st.markdown('<p class = "header">这是一个对基于房价数据,用线性回归,岭回归,拉索回归,决策数与随机森林的回归与分类,<br>'
            'AdaBoostRegressor,DecisionTreeRegressor,GradientBoostingRegressor,,<br>'
            '这些模型进行数据评估,最后分别得到每个模型的最佳模型,并可以对输入特征预测的app</p>',unsafe_allow_html=True)
#
# left,right = st.columns([0.5,0.5],gap='large')
# with left:
#   st.subheader('机器学习回归问题')
#   with st.expander('原数据'):
#     st.write('**Raw data**')
#     df = pd.read_csv('./链家二手房房源.csv')
#     df
#
#     st.write('**X**')
#     X_raw = df.drop('租价：元/月', axis=1)
#     X_raw
#
#     st.write('**y**')
#     y_raw = df['租价：元/月']
#     y_raw
#   with st.expander('模型的探索过程'):
#     st.write('不同模型的分数比较')
#     st.image('image/不同模型的比较.png')
#
# with right:
#   st.subheader('机器学习分类问题')
  # with st.expander('原数据'):


with st.expander('原数据'):
  st.write('**Raw data**')
  df = pd.read_csv('./链家二手房房源.csv')
  st.dataframe(df)

  st.write('**X**')
  X_raw = df.drop('租价：元/月', axis=1)
  st.dataframe(X_raw)

  st.write('**y**')
  y_raw = df['租价：元/月']
  st.dataframe(y_raw)
with st.expander("数据的基本可视化"):
    st.markdown("### 特征的基本情况")
    st.image('image/特征情况.png')
    col1, col2,col3 = st.columns(3)
    if col1.button("特征与区的关系",use_container_width=True):
        st.image('image/y与区的关系.png')
    if col2.button('特征与租赁方式的关系',use_container_width=True):
        st.image('image/y与租赁方式的关系.png')
    if col3.button('特征与官方核验的关系',use_container_width=True):
        st.image('image/y与官方核验的关系.png')
#输入要预测的特征
with st.sidebar:
  st.header('Input features')
  floor_whole = st.slider('总楼层数', data['总楼层数'].min(), data['总楼层数'].max() + 1, 10)
  rooms = st.slider('室数', data['室数'].min(), data['室数'].max() + 1, 3)
  halls = st.slider('厅数', data['厅数'].min(), data['厅数'].max() + 1, 3)
  guards = st.slider('卫数', data['卫数'].min(), data['卫数'].max() + 1, 3)
  lease_method = st.selectbox('租赁方式', ('合租','整租','第三人民医院家属院 3室2厅 南/北'))
  community = st.selectbox('小区',data0['区'].unique() )
  street = st.selectbox('街道',data0['街道'].unique())
  floor = st.selectbox('楼层',data0['楼层高低'].unique())
  towards = st.selectbox('朝向',data0['朝向'].unique())
  verified = 1 if  st.selectbox('官方是否核验', ('是','否')) else 0
  apartment = 1 if  st.selectbox('是否公寓', ('是','否')) else 0
  balcony = 1 if  st.selectbox('是否有独立阳台', ('是','否')) else 0
  CloseMetro = 1 if  st.selectbox('是否近地铁', ('是','否')) else 0
  SeparateToilet = 1 if  st.selectbox('是否有独立卫生间', ('是','否')) else 0
  CentralHeating = 1 if  st.selectbox('是否集中供暖', ('是','否')) else 0
  two_toilet =  1 if st.selectbox('是否双卫生间', ('是','否')) else 0
  Hardcover =  1 if  st.selectbox('是否精装', ('是','否')) else 0
  live =  1 if st.selectbox('是否拎包入住', ('是','否')) else 0
  is_new = 1 if st.selectbox('是否新上', ('是','否'))== '是' else 0
  month_rent = 1 if st.selectbox('是否月租', ('是','否')) else 0
  one_one = 1 if st.selectbox("是否押一付一",('是','否')) == '是' else 0
  anytime_view = 1 if st.selectbox("是否随时看房",('是','否')) == '是' else 0
#原始数据的展示
info0 = {
 '租赁方式': lease_method,
 '区': community,
 '街道': street,
 '朝向': towards,
 '户型': str(rooms) + '室' + str(halls) + '厅' + str(guards) + '卫',
 '楼层高低': floor,
 '总楼层数': floor_whole,
 '官方核验': "是" if verified == 1 else '否',
 '公寓': "是" if apartment == 1 else '否',
 '独立阳台':"是" if balcony == 1 else '否',
 '近地铁': "是" if CloseMetro == 1 else '否',
 '押一付一': "是" if one_one == 1 else '否',
 '独立卫生间': "是" if SeparateToilet == 1 else '否',
 '集中供暖': "是" if CentralHeating == 1 else '否',
 '双卫生间': "是" if two_toilet == 1 else '否',
 '精装': "是" if Hardcover == 1 else '否',
 '随时看房': "是" if apartment == 1 else '否',
 '拎包入住': "是" if live == 1 else '否',
 '新上': "是" if is_new == 1 else '否',
 '是否月租': "是" if month_rent == 1 else '否'
}
#将输入数据转化为供预测的形式
categories = ['区_临潼', '区_新城区', '区_未央', '区_灞桥', '区_碑林',
              '区_莲湖', '区_蓝田', '区_西咸新区（西安）', '区_鄠邑区', '区_长安',
              '区_雁塔', '区_高陵']
community_dict =   { category:category[2:] == community for category in categories}
category_lease = ['租赁方式_合租', '租赁方式_整租', '租赁方式_第三人民医院家属院 3室2厅 南/北']
lease_dict = {category:category[5:] == lease_method for category in category_lease}
category_floor = ['楼层高低_中楼层', '楼层高低_低楼层', '楼层高低_地下室', '楼层高低_高楼层']
floor_dict = { category:category[5:] == floor for category in category_floor}
lease_dict.update(community_dict)
lease_dict.update(floor_dict)
mappings = {
    '朝向': { v:k for k,v in dict(enumerate(data0['朝向'].astype('category').cat.categories)).items()},
    '街道':{ v:k for k,v in dict(enumerate(data0['街道'].astype('category').cat.categories)).items()},
}
street = mappings['街道'][street]
towards = mappings['朝向'][towards]

info = {
  '街道':street,
  '朝向':towards,
  '官方核验':verified,
  '公寓':apartment,
  '独立阳台':balcony,
  '近地铁':CloseMetro,
  '押一付一':one_one,
  '独立卫生间':SeparateToilet,
  '集中供暖':CentralHeating,
  '双卫生间':two_toilet,
  '精装':Hardcover,
  '随时看房':anytime_view,
  '拎包入住':live,
  '新上':is_new,
  '是否月租':month_rent,
  '室数':rooms,
  '厅数':halls,
  '卫数':guards,
  '总楼层数': floor_whole,
  # '租赁方式_合租': ,
  # '租赁方式_整租': None,
  # '租赁方式_第三人民医院家属院 3室2厅 南/北': None,
  # '区_临潼': None,
  # '区_新城区': None,
  # '区_未央': None,
  # '区_灞桥': None,
  # '区_碑林': None,
  # '区_莲湖': None,
  # '区_蓝田': None,
  # '区_西咸新区（西安）': None,
  # '区_鄠邑区': None,
  # '区_长安': None,
  # '区_雁塔': None,
  # '区_高陵': None,
  # '楼层高低_中楼层': None,
  # '楼层高低_低楼层': None,
  # '楼层高低_地下室': None,
  # '楼层高低_高楼层': None
}
info.update(lease_dict)
predict_data = pd.DataFrame(info,index=[0])
with st.expander('输入的特征集'):
  st.dataframe(pd.DataFrame(info0,index=[0]),)
#决策树,随机森林
with st.expander("线性回归、岭回归与拉索回归的探索过程"):
    left,right = st.columns(2)
    left.markdown("### 一元与多元线性回归")
    left.markdown("#### 实际数据与预测数据的散点分布")
    left.image('image/一元散点.png')
    left.markdown("#### 实际数据与预测数据的线性分布")
    left.image("image/一元线性.png")
    right.markdown("### 多元线性回归")
    right.markdown("#### 多元分类报告")
    right.image("image/多元混淆矩阵.png")
    right.markdown("image/多元线性分布")
    right.image("image/多远线性.png")
    st.markdown("### 岭回归与拉锁回归")
    left,right = st.columns(2)
    left.markdown("#### r2参数的比较")
    left.image("image/岭回归拉索回归r2.png")
    right.markdown('#### 岭回归的岭迹图')
    right.image("image/岭回归的岭迹图.png")
with st.expander('集成学习进阶的探索过程'):
  st.markdown("### 各个模型的交叉验证的比较图", unsafe_allow_html=True)
  st.image('image/不同模型的比较.png')
  st.markdown("### 各个模型的学习曲线", unsafe_allow_html=True)
  left,right = st.columns(2)
  left.markdown("#### GradientBoostingRegressor")
  left.image('image/GradientBoostingRegressor.png')
  right.markdown('#### AdaBoostRegressor')
  right.image('image/AdaBoostRegressor.png')
  left,right = st.columns(2)
  left.markdown("#### DecisionTreeRegressor")
  left.image('image/DecisionTreeRegressor.png')
  right.markdown("#### RandomForestRegressor")
  right.image('image/RandomForestRegressor.png')
  # model = st.selectbox('',('GradientBoostingRegressor','AdaBoostRegressor','DecisionTreeRegressor',"RandomForestRegressor"))
  # if model == 'GradientBoostingRegressor':
  #   st.image('image/GradientBoostingRegressor.png')
  # elif model == 'AdaBoostRegressor':
  #   st.image('image/AdaBoostRegressor.png')
  # elif model == 'DecisionTreeRegressor':
  #   st.image('image/DecisionTreeRegressor.png')
  # elif model == 'RandomForestRegressor':
  #   st.image('image/RandomForestRegressor.png')
  st.markdown("### 模型对输入特征的预测结果", unsafe_allow_html=True)
  left,middle,right = st.columns(3)
  left.markdown("#### ada模型预测结果")
  left.dataframe(joblib.load('models/adaboost_regressor_best_model.pkl').predict(predict_data))
  middle.markdown("#### gbdt模型的预测结果")
  middle.dataframe(joblib.load('models/gbdt_best_model.pkl').predict(predict_data))
  right.markdown("#### xgb模型的预测结果")
  right.dataframe(joblib.load('models/xgb_regressor_best_model.pkl').predict(predict_data))
with st.expander("决策树和随机森林回归角度的探索过程"):
  st.markdown("### 基于随机森林的探索", unsafe_allow_html=True)
  left,right = st.columns(2)
  # if left.button("各个特征重要性",use_container_width=True):
  #   left.image('image/各个特征累计重要性.png',width=600)
  # if right.button("特征累计重要性",use_container_width=True):
  #   right.image("image/随机森林特征累计重要性.png",width=600)
  left.markdown("#### 各个特征重要性", unsafe_allow_html=True)
  left.image('image/各个特征累计重要性.png',)
  right.write("特征累计重要性")
  right.image("image/随机森林特征累计重要性.png")
  st.markdown("#### 调参过程的得分情况", unsafe_allow_html=True)
  left,middle,right = st.columns(3)
  left.image('image/随机森林得分1.png')
  middle.image('image/随机森林得分2.png')
  right.image('image/随机森林得分3.png')
  st.markdown("### 决策树的探索", unsafe_allow_html=True)
  st.markdown("#### 叶节点可视化", unsafe_allow_html=True)
  left,middle,right = st.columns(3)
  left.image('image/决策树叶节点1.png')
  middle.image("image/决策树叶节点2.png")
  right.image("image/决策树叶节点3.png")
  st.markdown("#### 决策树分数图", unsafe_allow_html=True)
  left,right = st.columns(2)
  left.image("image/决策树分数1.png")
  right.image("image/决策树分数2.png")
  st.markdown("### 调参后模型对输入特征的预测", unsafe_allow_html=True)
  st.dataframe(predict_data)
  important_feature_name = ['总楼层数', '街道', '卫数', '朝向', '室数', '楼层高低_地下室',
                            '区_雁塔', '精装', '集中供暖', '公寓', '官方核验', '区_莲湖',
                            '近地铁', '楼层高低_低楼层', '厅数', '随时看房']
  left,right = st.columns(2)
  left.markdown("#### 随机森林的预测结果", unsafe_allow_html=True)
  left.dataframe(joblib.load('models/RandomForestRegressor.pkl').predict(predict_data[important_feature_name]))
  right.markdown('#### 决策树的预测结果', unsafe_allow_html=True)
  right.dataframe(joblib.load('models/DTR.pkl').predict(predict_data))
#基于不用模型的分析结果,然后每一个模型给一个调参的参数过程,
with st.expander("决策树和随机森林分类角度的探索过程"):
    st.markdown("### 基于决策树", unsafe_allow_html=True)
    st.markdown("#### 决策数评估过程的分数变化", unsafe_allow_html=True)
    left,middle1,middle2,right = st.columns(4)
    left.image("image/分类决策分数1.png")
    middle1.image("image/分类决策分数2.png")
    middle2.image("image/分类决策分数3.png")
    right.image("image/分类决策分数4.png")
    st.markdown("#### 决策树的c2评分报告", unsafe_allow_html=True)
    st.image("image/决策树评估报告.png")
    st.markdown("### 基于随机森林", unsafe_allow_html=True)
    st.markdown("#### 改变不同参数分数的变化", unsafe_allow_html=True)
    #原始参数,增多次数,不同树木个数的宽到细,深度,最大特征,min_samples_spli
    # 创建第一行的三列
    col1, col2, col3 = st.columns(3)
    col1.markdown("##### 原始参数的分数", unsafe_allow_html=True)
    col1.image('image/分类随机原始分数.png')
    col2.markdown("##### 多次循环的分数", unsafe_allow_html=True)
    col2.image("image/随机分类多次循环.png")
    col3.markdown('##### 不同树木的分数', unsafe_allow_html=True)
    col3.image("image/随机分类不同树木.png")
    col4,col5,col6 = st.columns(3)
    col4.markdown('##### 不同深度的分数', unsafe_allow_html=True)
    col4.image("image/随机分类不同深度.png")
    col5.markdown("##### 不同最大特征的分数", unsafe_allow_html=True)
    col5.image("image/随机分裂不同特征.png")
    col6.markdown("##### 不同最小分裂样本数的分数", unsafe_allow_html=True)
    col6.image("image/随机分类不同最小分割.png")
    st.markdown("#### 最佳模型的表现", unsafe_allow_html=True)
    st.markdown("##### 最佳模型的重要特征", unsafe_allow_html=True)
    st.image("image/分类随机最佳特征.png")
    st.markdown("##### 最佳模型的c2报告", unsafe_allow_html=True)
    st.image('image/分类随机最佳c2.png')

