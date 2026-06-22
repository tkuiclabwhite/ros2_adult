/*
 * Gamepad API Test
 * Written in 2013 by Ted Mielczarek <ted@mielczarek.org>
 *
 * To the extent possible under law, the author(s) have dedicated all copyright and related and neighboring rights to this software to the public domain worldwide. This software is distributed without any warranty.
 *
 * You should have received a copy of the CC0 Public Domain Dedication along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
 */
var buttonObj = new Object;//buttonNum, buttonValue
var buttonArr;
var axesArr;
function ReturnButtonValue()
{
	if(buttonObj.checkPressed){
		buttonObj.checkPressed = 0;
		buttonObj.check = 1;
		buttonObj.buttonArr = buttonArr;
		buttonObj.axesArr = axesArr;
		return buttonObj;
	}else{
		buttonObj.num = 0;
		buttonObj.val = 0;
		buttonObj.checkPressed = 0;
		buttonObj.check = 0;
		return buttonObj;
	}
}
var haveEvents = 'GamepadEvent' in window;
var haveWebkitEvents = 'WebKitGamepadEvent' in window;
var controllers = {};
var rAF = window.mozRequestAnimationFrame ||
  window.webkitRequestAnimationFrame ||
  window.requestAnimationFrame;

function connecthandler(e) {
  console.log("Gamepad connected at index %d: %s. %d buttons, %d axes.",
    e.gamepad.index, e.gamepad.id,
    e.gamepad.buttons.length, e.gamepad.axes.length);
  addgamepad(e.gamepad);
}
function addgamepad(gamepad) {
  controllers[gamepad.index] = gamepad;
  rAF(updateStatus);
}

function disconnecthandler(e) {
  console.log("Gamepad disconnected from index %d: %s",
    e.gamepad.index, e.gamepad.id);
  removegamepad(e.gamepad);
}

function removegamepad(gamepad) {
  delete controllers[gamepad.index];
}

function updateStatus() {
  scangamepads();
  for (j in controllers) {
    var controller = controllers[j];
    for (var i=0; i<controller.buttons.length; i++) {
      var val = controller.buttons[i];
      var pressed = val == 1.0;
      if (typeof(val) == "object") {
        pressed = val.pressed;
        val = val.value;
      }
			buttonArr[i] = val;
      if (pressed) {
				/*console.log("button "+i+" : "+val);*/
				buttonObj.num = i;
				buttonObj.val = val;
				buttonObj.checkPressed = 1;
      } else {
      }
    }

    for (var i=0; i<controller.axes.length; i++) {
			/*console.log(controller.axes[i]);*/
			if(Math.abs(controller.axes[i]) > 0.01) buttonObj.checkPressed = 1;
			axesArr = controller.axes;
    }
  }
  rAF(updateStatus);
}

function scangamepads() {
  var gamepads = navigator.getGamepads ? navigator.getGamepads() : (navigator.webkitGetGamepads ? navigator.webkitGetGamepads() : []);
  for (var i = 0; i < gamepads.length; i++) {
    if (gamepads[i]) {
			buttonObj.buttonLength = gamepads[i].buttons.length;
			buttonArr = new Array(buttonObj.buttonLength);
      if (!(gamepads[i].index in controllers)) {
        addgamepad(gamepads[i]);
      } else {
        controllers[gamepads[i].index] = gamepads[i];
      }
    }
  }
}

if (haveEvents) {
  window.addEventListener("gamepadconnected", connecthandler);
  window.addEventListener("gamepaddisconnected", disconnecthandler);
} else if (haveWebkitEvents) {
  window.addEventListener("webkitgamepadconnected", connecthandler);
  window.addEventListener("webkitgamepaddisconnected", disconnecthandler);
} else {
  setInterval(scangamepads, 500);
}
