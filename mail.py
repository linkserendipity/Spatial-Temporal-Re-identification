# coding=utf-8
import smtplib  
from email.mime.text import MIMEText  
from email.header import Header
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart  
import time
#邮件发送函数
#SMTPHost 邮件服务器
#fromAccount 发件邮箱
#fromPasswd 发件邮件授权码，注意并不是邮箱登录密码
#toAccount 收件邮箱
#subject 邮件标题
#content 邮件正文

def send_mail(content='what content?', SMTPHost='smtp.126.com', fromAccount='testgpu318@126.com', fromPasswd='YXSVONCFBRKATTTP', toAccount='testgpu318@126.com', subject='eval_acc'):    

    #构建邮件
    msg = MIMEMultipart()
    msg['Subject'] = Header(subject, 'utf-8')
    msg['From'] = fromAccount
    msg['To'] = toAccount

    # content = '<html><body><h3>主题内容</h3>' + '<p><a href="www.weatherfood.com">谭某人的博客</a></p>' + '</body></html>'
    puretext = MIMEText(content, 'html', 'utf-8')

    msg.attach(puretext)

    #! 附件
    # xlsxpart = MIMEApplication(open('tank/mask.txt', 'rb').read()) #这里填写你自己目录下的附件文件
    # xlsxpart.add_header('Content-Disposition', 'attachment', filename='tank/mask.txt')
    # msg.attach(xlsxpart)

    #使用smtplib模块发送邮件
    email_client = smtplib.SMTP(SMTPHost)
    # email_client.connect(SMTPHost, 25)
    email_client.login(fromAccount, fromPasswd)
    email_client.sendmail(fromAccount, toAccount, msg.as_string())
    email_client.quit()
    print('mail success')            



def get_gpu_memory():
    os.system('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free > tmp.txt')
    memory_gpu = [int(x.split()[2]) for x in open('tmp.txt', 'r').readlines()]
    os.system('rm tmp.txt')
    return memory_gpu


flag_last = get_gpu_memory() # 初始的GPU memory
gpu_number = len(flag_last)  # GPU个数
##############################################################################
time_circle = 1800            # 检测一次的周期(秒)    30min
iteration = 0                # 累计检测次数
max_num = 10                   # 发送邮件失败次数上限
##############################################################################
while True:
    
    # print(time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())) 
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) 
    gpu_memory = get_gpu_memory() #gpu_memory[0]
    print("gpu free memory:{} ".format(gpu_memory))

    #* test if GPU 0,1,2,3 is free
    num = 0 
    free_gpu_id = ''
    for i in range(gpu_number):
        if gpu_memory[i] > 31000:
            free_gpu_id = free_gpu_id + str(i) + ' '

    if len(free_gpu_id) > 0:
        print("GPU {}FREE! ".format(free_gpu_id)) #o(￣▽￣)o
        while True:
            try:
                send_mail("V100 GPU {}unused, free memory: {}".format(free_gpu_id, gpu_memory))
                # send_mail("test?????")
                break
            except:
                print('WWarning: email not sent.')
                num+=1
                if num>max_num:
                    print('发送邮件失败')
                    break
    else:
        print("No GPU is FREE! ") #ಠ_ಠ
        if iteration % 8 == 0:
            while True:
                try:
                    send_mail("All V100 GPUs are occupied, free memory: {}".format(gpu_memory))
                    # send_mail("test")
                    break
                except:
                    print('WWWWWarning: email not sent.')
                    num+=1
                    if num>max_num:
                        print('发送邮件失败')
                        break
        
    iteration = iteration + 1
    time.sleep(time_circle)

# if __name__ == '__main__':
#     acc=0.85
#     sendMail(content='result={}'.format(acc), toAccount='m18679060131@163.com',subject='t e s t')         
            
            # num1=0
            # while True:
            #     try:
            #         send_mail("auto_new.py:acc={}".format(acc))
            #         send_mail("auto_new.py:acc={}".format(acc), toAccount='testgpuinfo888@126.com')
            #         break
            #     except:
            #         print('WWarning: email not sent.')
            #         num1+=1
            #         if num1>10:
            #             print('mail failed')
            #             break      