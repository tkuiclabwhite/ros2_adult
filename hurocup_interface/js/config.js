window.ROBOT_CONFIG = {
    ip: '172.17.121.10'
};

document.addEventListener('DOMContentLoaded', function () {
    var sel = document.getElementById('addressSelect');
    if (!sel) return;
    var opt = document.createElement('option');
    opt.value = window.ROBOT_CONFIG.ip;
    opt.text  = window.ROBOT_CONFIG.ip;
    sel.insertBefore(opt, sel.firstChild);
    sel.value = window.ROBOT_CONFIG.ip;
});
