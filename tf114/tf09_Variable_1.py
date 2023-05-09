import tensorflow as tf
tf.compat.v1.set_random_seed(123)

변수 = tf.compat.v1.Variable(tf.random_normal([2]), name='weight') #random_normal이건 shape
print(변수) #<tf.Variable 'weight:0' shape=(1,) dtype=float32_ref>


#초기화 첫번째
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
aaa = sess.run(변수)
print('aaa : ',aaa) # aaa :  [-1.5080816   0.26086742]
sess.close()

#초기화 두번째
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
bbb = 변수.eval(session=sess) # 텐서플로 데이터형인  '변수'를 파이썬에서 볼수있는 놈으로 바꿔줘
print('bbb :', bbb) # bbb : [-1.5080816   0.26086742]
sess.close()

#초기화 세번째
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = 변수.eval() #InteractiveSession 안에 session=sess를 안넣어도됨.
print('ccc: ', ccc) #ccc:  [-1.5080816   0.26086742]
sess.close() 