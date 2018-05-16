# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import sys, os

import paddle.v2 as paddle

from resnet import resnet_cifar10
from MyReader import MyReader
with_gpu = os.getenv('WITH_GPU', '0') != '0'


def main():
    datadim = 3 * 32 * 32
    classdim = 100
    imageSize = 32
    myReader = MyReader(imageSize=imageSize)

    # PaddlePaddle init
    paddle.init(use_gpu=False, trainer_count=1)

    image = paddle.layer.data(
        name="image", type=paddle.data_type.dense_vector(datadim))

    # Add neural network config
    # option 1. resnet
    net = resnet_cifar10(image, depth=32)
    # option 2. vgg
    #net = vgg_bn_drop(image)

    out = paddle.layer.fc(
        input=net, size=classdim, act=paddle.activation.Softmax())

    lbl = paddle.layer.data(
        name="label", type=paddle.data_type.integer_value(classdim))
    cost = paddle.layer.classification_cost(input=out, label=lbl)

    # Create parameters
    parameters = paddle.parameters.create(cost)

    # Create optimizer
    momentum_optimizer = paddle.optimizer.Momentum(
        momentum=0.9,
        regularization=paddle.optimizer.L2Regularization(rate=0.0002 * 128),
        learning_rate=0.1 / 128.0,
        learning_rate_decay_a=0.1,
        learning_rate_decay_b=2725 * 100,
        learning_rate_schedule='discexp')

    # Create trainer
    trainer = paddle.trainer.SGD(
        cost=cost, parameters=parameters, update_equation=momentum_optimizer)

    # End batch and end pass event handler
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                print "\nPass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics)
            else:
                sys.stdout.write('.')
                sys.stdout.flush()
        if isinstance(event, paddle.event.EndPass):
            # save parameters
            with open('params_pass_%d.tar' % event.pass_id, 'w') as f:
                trainer.save_parameter_to_tar(f)
            test_reader = myReader.test_reader(test_list="../datasets/test.txt")
            result = trainer.test(
                reader=paddle.batch(
                    test_reader, batch_size=128),
                feeding={'image': 0,
                         'label': 1})
            print "\nTest with Pass %d, %s" % (event.pass_id, result.metrics)

    # Save the inference topology to protobuf.
    inference_topology = paddle.topology.Topology(layers=out)
    with open("inference_topology.pkl", 'wb') as f:
        inference_topology.serialize_for_inference(f)

    trainer_reader = myReader.train_reader(train_list="../datasets/train.txt")
    trainer.train(
        reader=paddle.batch(
            paddle.reader.shuffle(
                trainer_reader, buf_size=2725),
            batch_size=128),
        num_passes=200,
        event_handler=event_handler,
        feeding={'image': 0,
                 'label': 1})

    # inference
    from PIL import Image
    import numpy as np
    import os

    def load_image(file):
        im = Image.open(file)
        im = im.resize((32, 32), Image.ANTIALIAS)
        im = np.array(im).astype(np.float32)
        # The storage order of the loaded image is W(widht),
        # H(height), C(channel). PaddlePaddle requires
        # the CHW order, so transpose them.
        im = im.transpose((2, 0, 1))  # CHW
        # In the training phase, the channel order of CIFAR
        # image is B(Blue), G(green), R(Red). But PIL open
        # image in RGB mode. It must swap the channel order.
        im = im[(2, 1, 0), :, :]  # BGR
        im = im.flatten()
        im = im / 255.0
        return im

    test_data = []
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    test_data.append((load_image(cur_dir + '/datasets/train/0b7b02087bf40ad1b65bb73d5c2c11dfa8ecceb5.jpg'), ))

    # users can remove the comments and change the model name
    # with open('params_pass_50.tar', 'r') as f:
    #    parameters = paddle.parameters.Parameters.from_tar(f)

    probs = paddle.infer(
        output_layer=out, parameters=parameters, input=test_data)
    lab = np.argsort(-probs)  # probs and lab are the results of one batch data
    print "Label of image/0b7b02087bf40ad1b65bb73d5c2c11dfa8ecceb5.jpg(meibaolian) is: %d" % lab[0][0]


if __name__ == '__main__':
    main()
