##########현재 버전이 1.0이면 그냥 출력
##########현재 버전이 2.0이면 즉시실행모드를 끄고 출력
import tensorflow as tf
# print(tf.__version__) #1.14.0

# if tf.__version__[0]=='2':
#     tf.compat.v1.disable_eager_execution()
# print(tf.executing_eagerly())

########################## gpt #################################
print(f'현재 Tensorflow버전 : {tf.__version__}' )

if tf.__version__.startswith('1.'):
    print("현재 TensorFlow 버전: 1.0")
    # 추가로 수행할 작업이 있다면 이 부분에 작성하세요.

elif tf.__version__.startswith('2.'):
    print("현재 TensorFlow 버전: 2.0")
    tf.compat.v1.disable_eager_execution()
    # 추가로 수행할 작업이 있다면 이 부분에 작성하세요.

else:
    print("현재 TensorFlow 버전을 인식할 수 없습니다.")

