from Criterion import criterion
from Metric import accuracy
from Helper import disc_rank, drop_features2, drop_edges2

def train(feature, label, train_identifier, val_identifier, model, optimizer, epochs):
    for epoch in range(1, epochs + 1):
        model.train()

        optimizer.zero_grad()
        result = model(feature)
        train_pred = result[train_identifier]
        train_y = label[train_identifier]
        l2_loss = model.l2_regularization()
        loss = criterion(train_pred, train_y) + l2_loss
        acc = accuracy(train_pred, train_y)

        loss.backward()
        optimizer.step()

        model.eval()
        val_result = model(feature)
        val_pred = val_result[val_identifier]
        val_y = label[val_identifier]
        val_loss = criterion(val_pred, val_y)
        val_acc = accuracy(val_pred, val_y)

        if epoch % 10 == 0:
            print("Epoch {} - Train Acc : {}    Train Loss : {},    Val Acc : {}    Val Loss : {}".format(epoch, round(
                acc.item(), 2), round(loss.item(), 2), round(val_acc.item(), 2), round(val_loss.item(), 2)))


def GCA_test(model, feature, adj, label, test_identifier):
    model.eval()
    t1 = model(feature, adj)  # ,bias = True)

    t1 = model.projection(t1)
    tr1 = model.classification(t1)
    tr1_pred = tr1[test_identifier]
    tr1_y = label[test_identifier]
    # L2 regularization is not implemented yet

    tacc = accuracy(tr1_pred, tr1_y)

    print("Test Acc : {}".format(round(tacc.item(), 2)))

    return tr1, round(tacc.item(), 2)

def GCA_train(model, optimizer, feature, orig_adj, label, train_identifier, test_identifier, args, isdeap=False):
    #     save_path = args.model_save_path+'subject_dependent/'+date+'/'+sub_idx+'.pt'
    #     early_stopping = EarlyStopping(patience = args.patience, verbose = False, path=save_path)
    best_acc = 0
    best_epoch = 0
    best_model = None
    best_z = None
    #     w = 0.5

    rank = disc_rank(feature, label, train_identifier, args.out_channels)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        x1 = drop_features2(feature, rank, p=args.pf1, threshold=args.tpf1)
        x2 = drop_features2(feature, rank, p=args.pf2, threshold=args.tpf2)
        e1 = drop_edges(orig_adj, p=args.pe1, threshold=args.tpe1)
        e2 = drop_edges(orig_adj, p=args.pe2, threshold=args.tpe2)

        #         x1 = drop_features(feature, adj, p = 0.1, threshold = args.tpf1)
        #         x2 = drop_features(feature, adj, p = 0.2, threshold = args.tpf2)
        #         e1 = drop_edges(adj, p = 0.1, threshold = args.tpe1)
        #         e2 = drop_edges(adj, p = 0.2, threshold = args.tpe2)

        z1 = model(x1, e1)  # ,bias = True)
        #         z1 = model(feature,adj)
        z1 = model.projection(z1)
        z2 = model(x2, e2)
        z2 = model.projection(z2)

        #         ne1 = model.decoder(z1)
        #         ne2 = model.decoder(z2)

        #         ne1 = (ne1-ne1.min())/(ne1.max()-ne1.min())
        #         ne2 = (ne2-ne2.min())/(ne2.max()-ne2.min())
        #         nadj1 = w*adj + (1.-w)*ne1
        #         nadj2 = w*adj + (1.-w)*ne2
        #         nadj = 0.5*(nadj1+nadj2)
        #         print(nadj)

        r1 = model.classification(z1)
        r1_pred = r1[train_identifier]
        r1_y = label[train_identifier]
        # L2 regularization is not implemented yet
        labeled_loss1 = criterion(r1_pred, r1_y)
        r1_acc = accuracy(r1_pred, r1_y, isdeap)

        r2 = model.classification(z2)
        r2_pred = r2[train_identifier]
        r2_y = label[train_identifier]
        # L2 regularization is not implemented yet
        labeled_loss2 = criterion(r2_pred, r2_y)
        r2_acc = accuracy(r2_pred, r2_y, isdeap)

        contrastive_loss = model.loss(z1, z2)
        #         print(contrastive_loss)
        loss = (labeled_loss1 + labeled_loss2) / 2. + contrastive_loss * args.loss_lambda
        #         loss = labeled_loss1 + contrastive_loss*args.loss_lambda

        loss.backward()
        optimizer.step()

        #         orig_adj = nadj.detach().clone().to(device)
        #         print(orig_adj)
        #         adj = nadj.detach().clone().cuda()
        acc = (r1_acc + r2_acc) / 2.
        #         acc = r1_acc

        tr1_pred = r1[test_identifier]
        tr1_y = label[test_identifier]
        tr1_loss = criterion(tr1_pred, tr1_y)
        tr1_acc = accuracy(tr1_pred, tr1_y, isdeap)

        tr2_pred = r2[test_identifier]
        tr2_y = label[test_identifier]
        tr2_acc = accuracy(tr2_pred, tr2_y, isdeap)
        tr2_loss = criterion(tr2_pred, tr2_y)

        #         tr_acc = (tr1_acc + tr2_acc)/2.
        if tr1_acc > tr2_acc:
            result = r1
            tr_acc = tr1_acc
        else:
            result = r2
            tr_acc = tr2_acc

        tr_loss = (tr1_loss + tr2_loss) / 2.
        total_acc = (tr_acc + acc) / 2.

        if tr_acc > best_acc:
            best_acc = tr_acc
            best_epoch = epoch
            best_model = model

            best_result = result
            best_z = z1 if tr1_acc > tr2_acc else z2

        if epoch % 10 == 0:
            print(
                "Epoch {} - Train Acc : {}    Train Loss : {},    Test Acc : {},    Test Loss :{},    Total Acc : {}".format(
                    epoch, round(acc.item(), 2), round(loss.item(), 2), round(tr_acc.item(), 2),
                    round(tr_loss.item(), 2), round(total_acc.item(), 2)))

    #         early_stopping(vloss, model)
    #         if early_stopping.early_stop:
    #             print('Epoch : {} - Ealry Stopping'.format(epoch))
    #             break
    #     model.load_state_dict(torch.load(save_path))
    return model, best_acc, best_epoch, best_model, best_z, best_result


def GCA_train2(model, otimizer, feature, orig_adj, label, train_identifier, test_identifier, args, device, date=None,
               sub_idx=None, isdeap=False):
    #     save_path = args.model_save_path+'subject_dependent/'+date+'/'+sub_idx+'.pt'
    #     early_stopping = EarlyStopping(patience = args.patience, verbose = False, path=save_path)
    best_acc = 0
    best_epoch = 0
    best_model = None
    best_z = None
    #     w = 0.5

    rankf = disc_rank(feature, label, train_identifier, args.out_channels)
    rankf1 = rankf * args.pf1
    rankf2 = rankf * args.pf2
    ranke = edge_rank(adj)
    ranke1 = ranke * args.pe1
    ranke2 = ranke * args.pe2

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        x1 = drop_features2(rankf1, feature, threshold=args.tpf1)
        x2 = drop_features2(rankf2, feature, threshold=args.tpf2)
        e1 = drop_edges2(ranke1, orig_adj, threshold=args.tpe1)
        e2 = drop_edges2(ranke2, orig_adj, threshold=args.tpe2)

        #         x1 = drop_features(feature, adj, p = 0.1, threshold = args.tpf1)
        #         x2 = drop_features(feature, adj, p = 0.2, threshold = args.tpf2)
        #         e1 = drop_edges(adj, p = 0.1, threshold = args.tpe1)
        #         e2 = drop_edges(adj, p = 0.2, threshold = args.tpe2)

        z1 = model(x1, e1)  # ,bias = True)
        #         z1 = model(feature,adj)
        z1 = model.projection(z1)
        z2 = model(x2, e2)
        z2 = model.projection(z2)

        #         ne1 = model.decoder(z1)
        #         ne2 = model.decoder(z2)

        #         ne1 = (ne1-ne1.min())/(ne1.max()-ne1.min())
        #         ne2 = (ne2-ne2.min())/(ne2.max()-ne2.min())
        #         nadj1 = w*adj + (1.-w)*ne1
        #         nadj2 = w*adj + (1.-w)*ne2
        #         nadj = 0.5*(nadj1+nadj2)
        #         print(nadj)

        r1 = model.classification(z1)
        r1_pred = r1[train_identifier]
        r1_y = label[train_identifier]
        # L2 regularization is not implemented yet
        labeled_loss1 = criterion(r1_pred, r1_y)
        r1_acc = accuracy(r1_pred, r1_y, isdeap)

        r2 = model.classification(z2)
        r2_pred = r2[train_identifier]
        r2_y = label[train_identifier]
        # L2 regularization is not implemented yet
        labeled_loss2 = criterion(r2_pred, r2_y)
        r2_acc = accuracy(r2_pred, r2_y, isdeap)

        contrastive_loss = model.loss(z1, z2)
        #         print(contrastive_loss)
        loss = (labeled_loss1 + labeled_loss2) / 2. + contrastive_loss * args.loss_lambda
        #         loss = labeled_loss1 + contrastive_loss*args.loss_lambda

        loss.backward()
        optimizer.step()

        #         orig_adj = nadj.detach().clone().to(device)
        #         print(orig_adj)
        #         adj = nadj.detach().clone().cuda()
        acc = (r1_acc + r2_acc) / 2.
        #         acc = r1_acc

        tr1_pred = r1[test_identifier]
        tr1_y = label[test_identifier]
        tr1_loss = criterion(tr1_pred, tr1_y)
        tr1_acc = accuracy(tr1_pred, tr1_y, isdeap)

        tr2_pred = r2[test_identifier]
        tr2_y = label[test_identifier]
        tr2_acc = accuracy(tr2_pred, tr2_y, isdeap)
        tr2_loss = criterion(tr2_pred, tr2_y)

        #         tr_acc = (tr1_acc + tr2_acc)/2.
        if tr1_acc > tr2_acc:
            result = r1
            tr_acc = tr1_acc
        else:
            result = r2
            tr_acc = tr2_acc

        tr_loss = (tr1_loss + tr2_loss) / 2.
        total_acc = (tr_acc + acc) / 2.

        if tr_acc > best_acc:
            best_acc = tr_acc
            best_epoch = epoch
            best_model = model

            best_result = result
            best_z = z1 if tr1_acc > tr2_acc else z2

    #         if epoch % 50 == 0:
    #             print("Epoch {} - Train Acc : {}    Train Loss : {},    Test Acc : {},    Test Loss :{},    Total Acc : {}".format(epoch, round(acc.item(), 2), round(loss.item(),2), round(tr_acc.item(),2), round(tr_loss.item(),2), round(total_acc.item(), 2)))

    #         early_stopping(vloss, model)
    #         if early_stopping.early_stop:
    #             print('Epoch : {} - Ealry Stopping'.format(epoch))
    #             break
    #     model.load_state_dict(torch.load(save_path))
    return model, best_acc, best_epoch, best_model, best_z, best_result


def GTN_train(feature, adj, label, train_identifier, test_identifier, model, classifier, optimizer, epochs):
    best_acc = 0
    best_epoch = 0
    best_model = None
    best_z = None

    for epoch in range(1, epochs + 1):
        model.train()

        optimizer.zero_grad()
        out = model(feature, adj)
        result = classifier(out)

        train_pred = result[train_identifier]
        train_y = label[train_identifier]
        train_loss = criterion(train_pred, train_y)
        train_acc = accuracy(train_pred, train_y)

        train_loss.backward()
        optimizer.step()

        test_pred = result[test_identifier]
        test_y = label[test_identifier]
        test_loss = criterion(test_pred, test_y)
        test_acc = accuracy(test_pred, test_y)

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            best_model = model
            best_embedding = out

        if epoch % 10 == 0:
            print("Epoch {} - Train Acc : {}    Train Loss : {},    Test Acc : {},    Test Loss :{}".format(epoch,round(train_acc.item(), 2),
                                                                                                            round(train_loss.item(),2),
                                                                                                            round(test_acc.item(),2),
                                                                                                            round(test_loss.item(),2)))

    return model, best_acc, best_epoch, best_model, best_embedding, result
