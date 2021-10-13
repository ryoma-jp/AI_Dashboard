/* global event */
function save_active_tab(event) {
    let pre_activeTab = localStorage.getItem('activeTab');
    localStorage.setItem('activeTab', event.target.dataset['bsTarget']);

    let activeTab = localStorage.getItem('activeTab');
    console.log("[DEBUG] set active tab (" + pre_activeTab + " -> " + activeTab + ")");
}

(function () {
    console.log("[DEBUG] function() is called");
    let tabEl = document.querySelector('a[data-bs-target="#dataset"]');
    tabEl.addEventListener('show.bs.tab', function(event){save_active_tab(event)});

    tabEl = document.querySelector('a[data-bs-target="#training"]');
    tabEl.addEventListener('show.bs.tab', function(event){save_active_tab(event)});

    let activeTab = localStorage.getItem('activeTab');
    console.log("[DEBUG] activeTab: " + activeTab);
    if (activeTab) {
        let tabEl_active = document.querySelector('a[data-bs-target="' + activeTab + '"]');
        let tab = new bootstrap.Tab(tabEl_active);
        tab.show();
    }
})()

/* onchangeDataset: change dataset selection */
function onchangeDataset(id, value) {
    console.log("[DEBUG] onchangeDataset() is called");
    
    console.log("[DEBUG] id=" + id);
    console.log("[DEBUG] value=" + value);
    
    let select = document.getElementById(id);
    for (var i = 0; i < select.options.length; i++) {
        console.log("[DEBUG] select.options[" + i + "].value=" + select.options[i].value);
        if (select.options[i].value == value) {
            select.options[i].selected = true;
        } else {
            select.options[i].selected = false;
        }
    }
    
    let form = document.getElementById(id+"_form");
    form.submit();
}

