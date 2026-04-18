//Allows the ID to stick to the left side
$(window).load(function ()
{
  $('.box').scroll(function ()
  {
    $(this).find('.inthesmallbox2').css('left', $(this).scrollLeft());
  });
});

function NewMotionList()
{
  //creates new 'div' in the page
  var div1=document.createElement('div');
  div1.className = "inthesmallbox2";
  var div2=document.createElement('div');
  div2.className = "inthesmallbox4";

  //first column
  var input=document.createElement('input');
  input.type='text';
  input.className='textbox';
  input.style.backgroundColor='darkred';
  input.value=-1;
  div1.appendChild(input);

  //loop to create the rest of the 41 columns starting with 'Name'
  for (var i = 1; i <= 45; i++) 
  {  
    //odd number columns
    if (i%2==1 && i!=1) 
    {
      var input=document.createElement('input');
      input.type='text';
      input.className='textbox';
      input.value=0;
      div2.appendChild(input);
    }  
    //even numbers columns
    if (i%2==0) 
    {
      var input=document.createElement('input');
      input.type='text';
      input.className='textbox';
      input.value=0;
      div2.appendChild(input);
    }
    //second column
    if(i==1)
    {
      var input=document.createElement('input');
      input.type='text';
      input.className='textbox';
      input.value=-1;
      div2.appendChild(input);
    }
  }
  //appends it into the MotionTable <div> in MotionControlInterface.html
  document.getElementById('MotionTable').appendChild(div1);
  document.getElementById('MotionTable').appendChild(div2);
}

function NewRelativePosition()
{
  var num=document.getElementById('RelativePositionTable').getElementsByClassName('inthesmallbox2').length+1;
  
  //creates new 'div' in the page
  var div1=document.createElement('div');
  div1.className = "inthesmallbox2";
  var div2=document.createElement('div');
  div2.className = "inthesmallbox4";

  //first column    
  var input=document.createElement('input');
  input.type='text';
  input.className = 'textbox';
  input.style.backgroundColor='darkred';
  input.id = 'relativePosition'+num;
  input.value=-1;
  div1.appendChild(input);
      
  //loop to create the rest of the 41 columns starting with 'Name'
  for (var i = 1; i <= 41 ; i++) 
  {  
    //odd number columns
    if (i%2==1 && i!=1) 
    {
      var input=document.createElement('input');
      input.type='text';
      input.className='textbox';
      input.value=0;
      div2.appendChild(input);
    }  
    //even number columns
    if (i%2==0) 
    {
      var input=document.createElement('input');
      input.type='text';
      input.className='textbox';
      input.value=0;
      div2.appendChild(input);
    }
    //second column
    if(i==1)
    {
      var input=document.createElement('input');
      input.type='text';
      input.className='textbox';
      input.value=-1;
      div2.appendChild(input);
    }
  }  
  //appends them into the RelativePositionTable <div> in MotionControlInterface.html
  document.getElementById('RelativePositionTable').appendChild(div1);
  document.getElementById('RelativePositionTable').appendChild(div2);

  //sets the relativePosition ID to be the same value as relativeSpeed ID
  $('#relativePosition'+num).change(function ()
  {
    $('#relativeSpeed'+num).val($(this).val());
  });
}

function NewRelativeSpeed() {
  // 計算新列編號
  var num = document
    .getElementById('RelativeSpeedTable')
    .getElementsByClassName('inthesmallbox2')
    .length + 1;

  // 定義要用到的 specialCols
  var specialCols = [17, 18, 19, 20, 23, 24, 25, 26];

  // clamp helper
  function makeClamper(maxVal) {
    return function() {
      var v = parseFloat(this.value);
      if (!isNaN(v) && v > maxVal) {
        this.value = maxVal;
      }
    };
  }

  // 建立第一欄（relativeSpeedX），它不在 specialCols 裡，上限設 100
  var div1 = document.createElement('div');
  div1.className = "inthesmallbox2";
  var input1 = document.createElement('input');
  input1.type = 'text';
  input1.id = 'relativeSpeed' + num;
  input1.className = 'textbox';
  input1.style.backgroundColor = 'darkred';
  input1.value = -1;
  // clamp 限制
  // input1.addEventListener('change', makeClamper(100));
  div1.appendChild(input1);

  // 建立後面 41 欄
  var div2 = document.createElement('div');
  div2.className = "inthesmallbox4";

  // 用 let 來保證每個迴圈都能正確 capture i
  for (let i = 1; i <= 41; i++) {
    let input = document.createElement('input');
    input.type = 'text';
    input.className = 'textbox';

    // 設定初始值
    if (i === 1) {
      input.value = -1;
    } else if (specialCols.includes(i)) {
      input.value = 50;
    } else {
      input.value = 50;
    }

    // 決定這個欄位的 clamp 上限
    var maxVal = specialCols.includes(i) ? 10000 : 10000;
    input.addEventListener('change', makeClamper(maxVal));

    div2.appendChild(input);
  }

  // 插回畫面
  var table = document.getElementById('RelativeSpeedTable');
  table.appendChild(div1);
  table.appendChild(div2);

  // 綁定：當 relativeSpeedX 改變時，同步到 relativePositionX
  $('#relativeSpeed' + num).change(function () {
    $('#relativePosition' + num).val($(this).val());
  });
}



function NewAbsolutePosition()
{
  var num=document.getElementById('AbsolutePositionTable').getElementsByClassName('inthesmallbox2').length+1;  
  
  //creates new 'div' in the page
  var div1=document.createElement('div');
  div1.className = "inthesmallbox2";
  var div2=document.createElement('div');
  div2.className = "inthesmallbox4";

  //first column    
  var input=document.createElement('input');
  input.type='text';
  input.className='textbox';
  input.style.backgroundColor='darkred';
  input.id = 'absolutePosition'+num;
  input.value=-1;
  div1.appendChild(input);

  //loop to create the rest of the 41 columns starting with 'Name'
  for (var i = 1; i <= 41 ; i++) 
  {  
    //odd number columns
    if (i%2==1 && i!=1) 
    {
      var input=document.createElement('input');
      input.type='text';
      input.className='textbox';
      input.value=2048;
      div2.appendChild(input);
    }  
    //even number columns
    if (i%2==0) 
    {
      var input=document.createElement('input');
      input.type='text';
      input.className='textbox';
      input.value=2048;
      div2.appendChild(input);
    }
    //second column
    if(i==1)
    {
      var input=document.createElement('input');
      input.type='text';
      input.className='textbox';
      input.value=-1;
      div2.appendChild(input);
    }
  }
  //appends them into the AbsolutePositionTable <div> in the MotionControlInterface.html
  document.getElementById('AbsolutePositionTable').appendChild(div1);
  document.getElementById('AbsolutePositionTable').appendChild(div2);

  //sets the absolutePosition ID to be the same value as absoluteSpeed ID
  $('#absolutePosition'+num).change(function ()
  {
    $('#absoluteSpeed'+num).val($(this).val()); 
  });
}

// function NewAbsoluteSpeed()
// {
//   var num=document.getElementById('AbsoluteSpeedTable').getElementsByClassName('inthesmallbox2').length+1; 
  
//   //creates new 'div' in the page
//   var div1=document.createElement('div');
//   div1.className = "inthesmallbox2";
//   var div2=document.createElement('div');
//   div2.className = "inthesmallbox4";

//   //first column    
//   var input=document.createElement('input');
//   input.type='text';
//   input.className='textbox';
//   input.style.backgroundColor='darkred';
//   input.id = 'absoluteSpeed'+num;
//   input.value=-1;
//   div1.appendChild(input);

//   //loop to create the rest of the 41 columns starting with 'Name'
//   for (var i = 1; i <= 41 ; i++) 
//   {  
//     //odd number columns
//     if (i%2==1 && i!=1) 
//     {
//       var input=document.createElement('input');
//       input.type='text';
//       input.className='textbox';
//       input.value=10;
//       div2.appendChild(input);
//     }  
//     //even number columns
//     if (i%2==0) 
//     {
//       var input=document.createElement('input');
//       input.type='text';
//       input.className='textbox';
//       input.value=10;
//       div2.appendChild(input);
//     }
//     //second column
//     if(i==1)
//     {
//       var input=document.createElement('input');
//       input.type='text';
//       input.className='textbox';
//       input.value=-1;
//       div2.appendChild(input);
//     }
//   }
//   //appends them into AbsoluteSpeedTable <div> in the MotionControlInterface
//   document.getElementById('AbsoluteSpeedTable').appendChild(div1);
//   document.getElementById('AbsoluteSpeedTable').appendChild(div2);
  
//   //sets the absoluteSpeed ID to be the same value as absolutePosition ID
//   $('#absoluteSpeed'+num).change(function ()
//   {
//     $('#absolutePosition'+num).val($(this).val()); 
//   });
// }

function NewAbsoluteSpeed() {
  var num = document
    .getElementById('AbsoluteSpeedTable')
    .getElementsByClassName('inthesmallbox2')
    .length + 1;

  // 建立第一欄
  var div1 = document.createElement('div');
  div1.className = "inthesmallbox2";
  var input1 = document.createElement('input');
  input1.type = 'text';
  input1.className = 'textbox';
  input1.style.backgroundColor = 'darkred';
  input1.id = 'absoluteSpeed' + num;
  input1.value = -1;
  div1.appendChild(input1);

  // 建立後面 41 欄
  var div2 = document.createElement('div');
  div2.className = "inthesmallbox4";

  // 想要變 1000 的欄位 index（i 對應到第 2~42 欄中的第 i 欄）
  var specialCols = [17, 18, 19, 20, 23, 24, 25, 26];

  for (var i = 1; i <= 41; i++) {
    var input = document.createElement('input');
    input.type = 'text';
    input.className = 'textbox';

    if (i === 1) {
      // 原本第 2 欄 (i==1) 預設 -1
      input.value = -1;
    } else if (specialCols.includes(i)) {
      // 在 specialCols 裡面的，設為 1000
      input.value = 50;
    } else {
      // 其餘欄位維持預設 10
      input.value = 50;
    }

    div2.appendChild(input);
  }

  // 插入到畫面
  var table = document.getElementById('AbsoluteSpeedTable');
  table.appendChild(div1);
  table.appendChild(div2);

  // 綁定：#absoluteSpeedX 改變時，同步到 #absolutePositionX
  $('#absoluteSpeed' + num).change(function () {
    $('#absolutePosition' + num).val($(this).val());
  });
}

function Add()
{
  if(document.getElementById("MotionList").style.display == "initial")
  {
    NewMotionList();
  }
  else if(document.getElementById("RelativePosition").style.display == "initial" || document.getElementById("RelativeSpeed").style.display ==  "initial")
  {
    NewRelativePosition();
    NewRelativeSpeed();
  }
  else if(document.getElementById("AbsolutePosition").style.display == "initial" || document.getElementById("AbsoluteSpeed").style.display ==  "initial")
  {
    NewAbsolutePosition();
    NewAbsoluteSpeed();
  }
  document.getElementById('label').innerHTML = "Add is successful !!";
}

function Delete()
{
  var num = document.getElementById("chose_delete").value;
  var flag = false;
  if(document.getElementById("MotionList").style.display == "initial")
  {
    for(var i = 0; i < document.getElementById('MotionTable').getElementsByTagName('div').length; i += 2)
    {
      if(document.getElementById('MotionTable').getElementsByTagName('div')[i].getElementsByClassName('textbox')[0].value == num)
      {
        document.getElementById('MotionTable').removeChild(document.getElementById('MotionTable').getElementsByTagName('div')[i]);
        document.getElementById('MotionTable').removeChild(document.getElementById('MotionTable').getElementsByTagName('div')[i]);
        document.getElementById('label').innerHTML = "Delete is successful !!";
        flag = true;
        break;
      }
    }
  }
  else if(document.getElementById("RelativePosition").style.display == "initial" || document.getElementById("RelativeSpeed").style.display == "initial")
  {
    for(var i = 0;i < document.getElementById('RelativePositionTable').getElementsByTagName('div').length;i += 2)
	  {
      if(document.getElementById('RelativePositionTable').getElementsByTagName('div')[i].getElementsByClassName('textbox')[0].value == num)
	    {
        document.getElementById('RelativePositionTable').removeChild(document.getElementById('RelativePositionTable').getElementsByTagName('div')[i]);
        document.getElementById('RelativePositionTable').removeChild(document.getElementById('RelativePositionTable').getElementsByTagName('div')[i]);
        document.getElementById('RelativeSpeedTable').removeChild(document.getElementById('RelativeSpeedTable').getElementsByTagName('div')[i]);
        document.getElementById('RelativeSpeedTable').removeChild(document.getElementById('RelativeSpeedTable').getElementsByTagName('div')[i]);
        document.getElementById('label').innerHTML = "Delete is successful !!";
        flag = true;
        break;
      }
    }
  }
  else if(document.getElementById("AbsolutePosition").style.display == "initial" || document.getElementById("AbsoluteSpeed").style.display ==  "initial")
  {
    for(var i = 0; i < document.getElementById('AbsolutePositionTable').getElementsByTagName('div').length; i += 2)
	  {
      if(document.getElementById('AbsolutePositionTable').getElementsByTagName('div')[i].getElementsByClassName('textbox')[0].value == num)
	    {
        document.getElementById('AbsolutePositionTable').removeChild(document.getElementById('AbsolutePositionTable').getElementsByTagName('div')[i]);
        document.getElementById('AbsolutePositionTable').removeChild(document.getElementById('AbsolutePositionTable').getElementsByTagName('div')[i]);
        document.getElementById('AbsoluteSpeedTable').removeChild(document.getElementById('AbsoluteSpeedTable').getElementsByTagName('div')[i]);
        document.getElementById('AbsoluteSpeedTable').removeChild(document.getElementById('AbsoluteSpeedTable').getElementsByTagName('div')[i]);
        document.getElementById('label').innerHTML = "Delete is successful !!";
        flag = true;
        break;
      }
    }
  }
  if(flag == false)
  {
	  document.getElementById('label').innerHTML = "Delete is fail !! No this ID!!";
  }
}

function Reverse()
{
  var num = document.getElementById("chose_reverse").value;
  var flag = false;
  var n = 0;
  if(document.getElementById("MotionList").style.display == "initial")
  {
    document.getElementById('label').innerHTML = "Reverse can only execute in RelativePosition!!";
  }
  else if(document.getElementById("RelativePosition").style.display == "initial")
  {
    for(var i = 0; i < document.getElementById('RelativePositionTable').getElementsByTagName('div').length; i += 2)
	  {
      if(document.getElementById('RelativePositionTable').getElementsByTagName('div')[i].getElementsByClassName('textbox')[0].value==num){
        n = i;
        flag = true;
        break;
      }  
    }
    if(flag == true)
	  {
      for(var j = 1; j <= 29; j++)
	    {
        var value = Number(document.getElementById('RelativePositionTable').getElementsByTagName('div')[n+1].getElementsByClassName('textbox')[j].value);
        document.getElementById('RelativePositionTable').getElementsByTagName('div')[n+1].getElementsByClassName('textbox')[j].value = value * (-1);
      }
      document.getElementById('label').innerHTML = "Reverse is successful !!";
    }
    else
	  {
      document.getElementById('label').innerHTML = "Reverse is fail !! No this ID !!";
    }
    
  }
  else if(document.getElementById("RelativeSpeed").style.display == "initial")
  {
    document.getElementById('label').innerHTML = "Reverse can only execute in RelativePosition!!"; 
  }
  else if(document.getElementById("AbsolutePosition").style.display == "initial")
  {
    document.getElementById('label').innerHTML = "Reverse can only execute in RelativePosition!!";
  }
  else if(document.getElementById("AbsoluteSpeed").style.display == "initial")
  {
    document.getElementById('label').innerHTML = "Reverse can only execute in RelativePosition!!";
  }
}

function Copy()
{
  var num = document.getElementById("chose_copy").value;
  var flag = false;
  var n = 0;
  if(document.getElementById("MotionList").style.display == "initial")
  {
    document.getElementById('label').innerHTML = "Copy can only execute in Relative!!";
  }
  else if(document.getElementById("RelativePosition").style.display == "initial" || document.getElementById("RelativeSpeed").style.display == "initial")
  {
    for(var i = 0; i < document.getElementById('RelativePositionTable').getElementsByTagName('div').length; i += 2)
	  {
      if(document.getElementById('RelativePositionTable').getElementsByTagName('div')[i].getElementsByClassName('textbox')[0].value == num)
	    {
        n = i;
        flag = true;
        break;
      }  
    }
    if(flag == true)
	  {
      Add();
      var num = document.getElementById('RelativePositionTable').getElementsByTagName('div').length;
      for (var j = 1; j <= 27; j++)
	    {

        var x = Number(document.getElementById('RelativePositionTable').getElementsByTagName('div')[n+1].getElementsByClassName('textbox')[j].value);
        var y = Number(document.getElementById('RelativeSpeedTable').getElementsByTagName('div')[n+1].getElementsByClassName('textbox')[j].value);
        document.getElementById('RelativePositionTable').getElementsByTagName('div')[num-1].getElementsByClassName('textbox')[j].value = x;
        document.getElementById('RelativeSpeedTable').getElementsByTagName('div')[num-1].getElementsByClassName('textbox')[j].value = y;
      }
      document.getElementById('label').innerHTML = "Copy is successful !!";
    }
    else
	  {
      document.getElementById('label').innerHTML = "Copy is fail !! No this ID !!";
    }
  }
  else if(document.getElementById("AbsolutePosition").style.display == "initial")
  {
    document.getElementById('label').innerHTML = "Copy can only execute in Relative!!";
  }
  else if(document.getElementById("AbsoluteSpeed").style.display == "initial")
  {
    document.getElementById('label').innerHTML = "Copy can only execute in Relative!!";
  }
}

function CopyToAbsolute()
{
  // 1. 自動切換到 Absolute 模式 (保持 UI 同步)
  MotionList(3);
  document.getElementsByName("List")[3].checked = true;

  // 2. 觸發新增一行
  Add();

  // 3. 取得剛產生的最後一列
  var tablePos = document.getElementById('AbsolutePositionTable');
  var tableSpd = document.getElementById('AbsoluteSpeedTable');
  var totalRows = tablePos.getElementsByTagName('div').length; 

  // 陣列中 totalRows - 1 就是最新新增的那組資料列 (inthesmallbox4)
  var lastRowPos = tablePos.getElementsByTagName('div')[totalRows - 1];
  var lastRowSpd = tableSpd.getElementsByTagName('div')[totalRows - 1];

  // 4. 迴圈讀取 motor_val_1 ~ 21 的數值並寫入
const customDefaults = {
    16:  { pos: 151912, spd: 1000 },
    17: { pos: 151912,  spd: 1000 },
    18:  { pos: 151912, spd: 1000 },  
    19:  { pos: 151912, spd: 1000 },
    22:  { pos: 151912, spd: 1000 },
    23:  { pos: 151912, spd: 1000 },
    24:  { pos: 151912, spd: 1000 },
    25:  { pos: 151912, spd: 1000 },
  };

  // 4. 迴圈讀取 motor_val_1 ~ 27 的數值並寫入
  for (var i = 1; i <= 27; i++) 
  {
    // 先抓取 HTML 元素 (防呆：確保該欄位真的存在)
    var motorElement = document.getElementById('motorval' + i);
    var motorValue = motorElement ? Number(motorElement.value) : 0;

    // 決定這顆馬達的預設值與速度
    // 如果 customDefaults 有設定就用設定的，沒有的話：位置給 2048，速度給 50
    var fallbackPos = (customDefaults[i] && customDefaults[i].pos !== undefined) ? customDefaults[i].pos : 2048;
    var targetSpd   = (customDefaults[i] && customDefaults[i].spd !== undefined) ? customDefaults[i].spd : 50;

    // 將數值填入 AbsolutePosition (如果有讀到有效數值且不為 0 就用讀到的，否則用 fallbackPos)
    lastRowPos.getElementsByClassName('textbox')[i].value = (motorValue !== 0) ? motorValue : fallbackPos;

    // 將數值填入 AbsoluteSpeed (直接套用上面決定的速度)
    lastRowSpd.getElementsByClassName('textbox')[i].value = targetSpd;
  }

  document.getElementById('label').innerHTML = "Motor values copied to Absolute successfully !!";
}

function CheckSum()
{
  var ID = Number(document.getElementById('CheckSumID').value);
  var flag = false;
  if(flag == false)
  {
	  for (var i = 0 ; i < document.getElementById('MotionTable').getElementsByTagName('div').length; i += 2) 
  	{
      if (document.getElementById('MotionTable').getElementsByTagName('div')[i].getElementsByClassName('textbox')[0].value == ID) 
      {
        var Sum = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        for (var j = 1; j <= (20 * 2); j += 2) 
        {
          if (Number(document.getElementById('MotionTable').getElementsByTagName('div')[i+1].getElementsByClassName('textbox')[j].value)) 
          {
            for (var k = 0; k < document.getElementById('RelativePositionTable').getElementsByTagName('div').length; k += 2) 
            {
			        Number(Number(document.getElementById('RelativePositionTable').getElementsByTagName('div')[k].getElementsByClassName('textbox')[0].value));
              if (Number(document.getElementById('MotionTable').getElementsByTagName('div')[i+1].getElementsByClassName('textbox')[j].value) == Number(document.getElementById('RelativePositionTable').getElementsByTagName('div')[k].getElementsByClassName('textbox')[0].value)) 
              {
                for (var l = 1; l <= 25; l++) 
                {
                  Sum[l-1] += (Number(document.getElementById('RelativePositionTable').getElementsByTagName('div')[k+1].getElementsByClassName('textbox')[l].value));
				        }
			        }
			      }
		      }
		    }     
        for (var j = 1; j <= 25; j++) 
        {
            document.getElementById("CheckSumBox").getElementsByTagName("div")[j-1].innerHTML = "";
            document.getElementById("CheckSumBox").getElementsByTagName("div")[j-1].innerHTML = "M" + j + ": " + Sum[j-1];
        }
		    flag = true;
        break;
      }
    }
  }
  if(flag == false)
  {
    for (var i = 0; i < document.getElementById('RelativePositionTable').getElementsByTagName('div').length; i += 2) 
    {
      if (document.getElementById('RelativePositionTable').getElementsByTagName('div')[i].getElementsByClassName('textbox')[0].value == ID) 
      {
        for (var j = 1; j <= 25; j++) 
        {
          document.getElementById("CheckSumBox").getElementsByTagName("div")[j-1].innerHTML = "";
          document.getElementById("CheckSumBox").getElementsByTagName("div")[j-1].innerHTML = "M" + j + ": " + (Number(document.getElementById('RelativePositionTable').getElementsByTagName('div')[i+1].getElementsByClassName('textbox')[j].value));
        }
		    flag = true;
        break;
      }
    }
  }
  if(flag == false)
  {
    for (var i = 0; i < document.getElementById('AbsolutePositionTable').getElementsByTagName('div').length; i += 2) 
    {
      if (document.getElementById('AbsolutePositionTable').getElementsByTagName('div')[i].getElementsByClassName('textbox')[0].value == ID) 
      {
        for (var j = 1; j <= 25; j++) 
        {
          document.getElementById("CheckSumBox").getElementsByTagName("div")[j-1].innerHTML = "";
          document.getElementById("CheckSumBox").getElementsByTagName("div")[j-1].innerHTML = "M" + j + ": " + (Number(document.getElementById('AbsolutePositionTable').getElementsByTagName('div')[i+1].getElementsByClassName('textbox')[j].value));
        }
		    flag = true;
        break;
      }
    }
  }
  if(flag == true)
  {
    document.getElementById('label').innerHTML = "CheckSum is successful !!";
  }
  else
  {
    document.getElementById('label').innerHTML = "CheckSum is fail !! No this ID !!";
  }
}

function MotionList(mode)
{
  switch(Number(mode))
  {
    case 0:
      document.getElementById("MotionList").style.display = "initial";
      document.getElementById("RelativePosition").style.display = "none";
      document.getElementById("RelativeSpeed").style.display = "none";
      document.getElementById("AbsolutePosition").style.display = "none";
      document.getElementById("AbsoluteSpeed").style.display = "none";
      break;
    case 1:
      document.getElementById("MotionList").style.display = "none";
      document.getElementById("RelativePosition").style.display = "initial";
      document.getElementById("RelativeSpeed").style.display = "none";
      document.getElementById("AbsolutePosition").style.display = "none";
      document.getElementById("AbsoluteSpeed").style.display = "none";
      break;
    case 2:
      document.getElementById("MotionList").style.display = "none";
      document.getElementById("RelativePosition").style.display = "none";
      document.getElementById("RelativeSpeed").style.display = "initial";
      document.getElementById("AbsolutePosition").style.display = "none";
      document.getElementById("AbsoluteSpeed").style.display = "none";
      break;
    case 3:
      document.getElementById("MotionList").style.display = "none";
      document.getElementById("RelativePosition").style.display = "none";
      document.getElementById("RelativeSpeed").style.display = "none";
      document.getElementById("AbsolutePosition").style.display = "initial";
      document.getElementById("AbsoluteSpeed").style.display = "none";
      break;
    case 4:
      document.getElementById("MotionList").style.display = "none";
      document.getElementById("RelativePosition").style.display = "none";
      document.getElementById("RelativeSpeed").style.display = "none";
      document.getElementById("AbsolutePosition").style.display = "none";
      document.getElementById("AbsoluteSpeed").style.display = "initial";
      break;
    }
}
