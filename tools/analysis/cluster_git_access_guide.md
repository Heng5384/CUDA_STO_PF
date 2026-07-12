# 在你的服务器上使用这个仓库

这份说明是写给“第一次在自己服务器上拿这个仓库的人”的。

你的目标通常只有两件事：

1. 在自己的服务器上拿到这个仓库
2. 后续能稳定地 `git pull` 更新，并跑 workflow

如果仓库是私有仓库，你不能直接假设自己有权限。正确顺序是：

1. 先申请权限
2. 再在服务器上配置 GitHub SSH 访问
3. 然后 `git clone`
4. 后续用 `git pull --ff-only origin main` 更新

---

## 1. 你需要先准备什么

在开始之前，你只需要确认下面几件事：

- 你有一台自己的服务器或 cluster 账户
- 这台机器可以访问 GitHub
- 你知道自己想把仓库放在哪个目录
- 你可以把自己的 GitHub 用户名或者服务器 SSH 公钥发给仓库维护者

---

## 2. 先向仓库维护者申请权限

如果你还没有权限，不要先尝试 `git clone`。

先把下面这段发给仓库维护者即可：

```text
我需要在自己的服务器上使用 CUDA_STO_PF 仓库。

请给我开通访问权限。我提供的信息如下：
1. GitHub 用户名：<your_github_username>
2. 服务器用途：cluster / workstation
3. 需要的权限：read-only / read-write
4. 如果你希望按服务器公钥授权，我可以再发 SSH 公钥
```

如果维护者要求你提供服务器 SSH 公钥，就继续看下一节。

---

## 3. 在服务器上生成 SSH 公钥

登录到你的服务器后执行：

```bash
ssh-keygen -t ed25519 -C "github-access-for-CUDA_STO_PF"
```

默认文件位置通常是：

- 私钥：
  - `~/.ssh/id_ed25519`
- 公钥：
  - `~/.ssh/id_ed25519.pub`

查看公钥内容：

```bash
cat ~/.ssh/id_ed25519.pub
```

把输出内容完整发给仓库维护者。

你可以直接发下面这个模板：

```text
我需要在自己的服务器上访问 CUDA_STO_PF 仓库。

GitHub 用户名：<your_github_username>
服务器用途：cluster
权限需求：read-only
SSH 公钥：
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI... your_name@your_server
```

---

## 4. 权限开通后，先测试 GitHub SSH

在服务器上执行：

```bash
ssh -T git@github.com
```

如果权限和 key 都正常，通常会看到类似：

```text
Hi <username>! You've successfully authenticated, but GitHub does not provide shell access.
```

如果失败，先检查：

- 维护者是否真的已经加了权限
- 服务器是否用了你刚才生成的 SSH key
- `~/.ssh/config` 是否需要显式指定 key

---

## 5. 如果需要，给服务器加一个简单的 GitHub SSH 配置

如果你的服务器不会自动选对 key，可以在：

- `~/.ssh/config`

里加：

```sshconfig
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
```

然后再测试一次：

```bash
ssh -T git@github.com
```

---

## 6. 第一次 clone 仓库

假设你准备把仓库放在：

- `/data/home/<your_user>/CUDA_STO_PF`

那么：

```bash
cd /data/home/<your_user>
git clone git@github.com:Heng5384/CUDA_STO_PF.git
cd CUDA_STO_PF
```

clone 完以后，建议马上确认一下：

```bash
git branch --show-current
git log --oneline -n 3
```

---

## 7. 以后如何更新到最新版本

进入仓库目录后执行：

```bash
cd /data/home/<your_user>/CUDA_STO_PF
git pull --ff-only origin main
```

推荐始终用这条，而不是直接 `git pull`。

原因很简单：

- `--ff-only` 更安全
- 如果你本地有改动，它会直接停下来提醒你
- 不会偷偷做自动 merge

---

## 8. 如果 `git pull` 失败怎么办

先看你的工作区是不是脏的：

```bash
git status --short
```

如果你只是临时改过一些文件，可以先暂存起来：

```bash
git stash push -u -m "pre-pull-backup"
git pull --ff-only origin main
```

如果之后你还想把这些改动拿回来：

```bash
git stash list
git stash pop
```

---

## 9. 在你的服务器上开始跑标准 workflow

建议你先设 3 个变量：

```bash
export REPO_ROOT=/data/home/<your_user>/CUDA_STO_PF
export QUEUE=gpu_uvip
export QOS=gpu_uvip
```

如果你的服务器队列名字不同，就把：

- `QUEUE`
- `QOS`

改成你自己的即可。

之后所有命令都统一从：

```bash
cd "${REPO_ROOT}"
```

开始。

完整 workflow 说明看这里：

- [critical_radius_continue_workflow.md](critical_radius_continue_workflow.md)

---

## 10. 你实际只需要改哪几项

如果你已经拿到了仓库权限，那么后面真正需要你自己改的，一般只有这几项：

1. `REPO_ROOT`
   - 你的仓库绝对路径
2. `QUEUE`
   - 你服务器上可用的 partition 名称
3. `QOS`
   - 对应的 qos 名称

其余 workflow 命令一般不需要改代码。

---

## 11. 最短上手清单

如果你只是想最快开始，按这个顺序做就行：

### 第一步：向维护者申请权限

```text
我需要在自己的服务器上使用 CUDA_STO_PF 仓库。

GitHub 用户名：<your_github_username>
服务器用途：cluster
权限需求：read-only
如需要，我可以提供服务器 SSH 公钥。
```

### 第二步：如果被要求，生成并发送 SSH 公钥

```bash
ssh-keygen -t ed25519 -C "github-access-for-CUDA_STO_PF"
cat ~/.ssh/id_ed25519.pub
```

### 第三步：测试 GitHub 连接

```bash
ssh -T git@github.com
```

### 第四步：clone 仓库

```bash
cd /data/home/<your_user>
git clone git@github.com:Heng5384/CUDA_STO_PF.git
cd CUDA_STO_PF
```

### 第五步：以后更新

```bash
cd /data/home/<your_user>/CUDA_STO_PF
git pull --ff-only origin main
```

### 第六步：开始 workflow

```bash
export REPO_ROOT=/data/home/<your_user>/CUDA_STO_PF
export QUEUE=gpu_uvip
export QOS=gpu_uvip

cd "${REPO_ROOT}"
```

然后按 workflow 文档执行：

- [critical_radius_continue_workflow.md](critical_radius_continue_workflow.md)
