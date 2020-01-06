import tensorflow as tf


class CrossStitch(tf.keras.layers.Layer):

    """Cross-Stitch implementation according to arXiv:1604.03539
    Implementation adapted from https://github.com/helloyide/Cross-stitch-Networks-for-Multi-task-Learning"""

    def __init__(self, num_tasks, *args, **kwargs):
        """initialize class variables"""
        self.num_tasks = num_tasks
        super(CrossStitch, self).__init__(**kwargs)

    def build(self, input_shape):
        """initialize the kernel and set the instance to 'built'"""
        self.kernel = self.add_weight(name="kernel",
                                      shape=(self.num_tasks,
                                             self.num_tasks),
                                      initializer='identity',
                                      trainable=True)
        super(CrossStitch, self).build(input_shape)

    def call(self, xl):
        """
        called by TensorFlow when the model gets build. 
        Returns a stacked tensor with num_tasks channels in the 0 dimension, 
        which need to be unstacked.
        """
        if (len(xl) != self.num_tasks):
            # should not happen
            raise ValueError()

        out_values = []
        for this_task in range(self.num_tasks):
            this_weight = self.kernel[this_task, this_task]
            out = tf.math.scalar_mul(this_weight, xl[this_task])
            for other_task in range(self.num_tasks):
                if this_task == other_task:
                    continue  # already weighted!
                other_weight = self.kernel[this_task, other_task]
                out += tf.math.scalar_mul(other_weight, xl[other_task])
            out_values.append(out)
        # HACK!
        # unless we stack, and then unstack the tensors, TF (2.0.0) can't follow
        # the graph, so it aborts during model initialization.
        return tf.stack(out_values, axis=0)

    def compute_output_shape(self, input_shape):
        return [self.num_tasks] + input_shape

    def get_config(self):
        """implemented so keras can save the model to json/yml"""
        config = {
            "num_tasks": self.num_tasks
        }
        base_config = super(CrossStitch, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))


if __name__ == "__main__":

    inputs = tf.keras.layers.Input(shape=[27, 107, 50])

    num_tasks = 2
    tops = [inputs] * num_tasks
    for task_id in range(num_tasks):
        in_tensor = tops[task_id]
        conv = tf.keras.layers.Conv2D(
            filters=3,
            kernel_size=(10, 1),
        )(in_tensor)
        tops[task_id] = conv

    cs = CrossStitch(num_tasks)(tops)
    tops = tf.unstack(cs, axis=0)

    model = tf.keras.Model(inputs=inputs, outputs=tops)
    model.summary()
