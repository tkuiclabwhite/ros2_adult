const rclnodejs = require('rclnodejs');

rclnodejs.init().then(() => {
  const node = rclnodejs.createNode('my_node');
  const parameter = node.getParameter('/location');
  console.log('/location parameter value:', parameter.value);

  const client = node.createClient('tku_msgs/srv/SaveHSV', '/SaveHSV');
  const request = { save: true, location: parameter.value };

  client.sendRequest(request, (response) => {
    console.log('Service response:', response);
  });

  rclnodejs.spin(node);
}).catch(console.error);