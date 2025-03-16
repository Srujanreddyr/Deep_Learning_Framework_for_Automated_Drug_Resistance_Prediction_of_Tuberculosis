from django.shortcuts import render, HttpResponse
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.core.files.storage import FileSystemStorage
from matplotlib import pyplot as plt


# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHome.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHome.html', {})


def TuberculosisTraining(request):
    from .utility.StartTraining import tuberculosis_training_start
    history, val_acc, val_loss,metrcis = tuberculosis_training_start()
    mylist = zip(history.get('loss'), history.get('accuracy'),history.get('val_loss'),history.get('val_accuracy'),history.get('lr'))
    return render(request, 'users/UserTraining.html',
                  {'history': mylist, 'val_acc': val_acc, 'val_loss': val_loss,'metrcis': metrcis})


def ConventionalTraining(request):
    from .utility.ConventionalApproach import tuberculosis_training_start
    history, val_acc, val_loss, metrcis = tuberculosis_training_start()
    mylist = zip(history.get('loss'), history.get('accuracy'), history.get('val_loss'), history.get('val_accuracy'),
                 history.get('lr'))
    return render(request, 'users/ConventionalTraining.html',
                  {'history': mylist, 'val_acc': val_acc, 'val_loss': val_loss, 'metrcis': metrcis})


def TestMyModel(request):
    if request.method == 'POST':
        myfile = request.FILES['file']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        from .utility import TestDRDS_Model
        result = TestDRDS_Model.start_test(filename)
        print('Result:', result)
        return render(request, "users/testform.html", {"result": result, "path": uploaded_file_url})
    else:
        return render(request, "users/testform.html", {})