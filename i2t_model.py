import torch
import torch.nn as nn
import clip
import torch.nn.functional as F
import MHTransformer
from utils import get_img_patch_feats, contrastive_loss, token_replace, prompt_token_replace, encode_text_img_learnable, contrastive_loss2
from third_party.open_clip.clip import tokenize

class Phi(nn.Module):
    """
    Textual Inversion Phi network.
    Takes as input the visual features of an image and outputs the pseudo-word token embedding.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)


class IMG2TEXT(nn.Module):
    """
        Fine-grained Textual Inversion network, we simply named IMG2TEXT.
    """

    def __init__(self, args, clip_model, img_patch_dim, token_feat, phi_s, phi_a, num_k, hy_regLoss, temperature, tf_layer,
                 tf_head, topk, epsilon, transformer):
        super().__init__()
        ## edit 
        self.args = args
        dtype = clip_model.dtype
        prompt_dim = clip_model.ln_final.weight.shape[0]
        if args.meta_prompt and args.n_ctx > 0:
            # 20240224
            if args.date == '20240224':
                """shred meta net weight"""
                self.meta_net = nn.Sequential(nn.Linear(prompt_dim, prompt_dim * 2), nn.ReLU(), nn.Dropout(0.5),
                                            nn.Linear(prompt_dim * 2, prompt_dim * 4), nn.ReLU(), nn.Dropout(0.5),
                                            nn.Linear(prompt_dim * 4, prompt_dim * 4), nn.ReLU(), nn.Dropout(0.5),
                                            nn.Linear(prompt_dim * 4, prompt_dim), nn.ReLU(), nn.Dropout(0.5))
            
            # 20240227
            elif args.date == '20240227':
                self.meta_net0 = nn.Sequential(nn.Linear(prompt_dim, prompt_dim * 2), nn.ReLU(), nn.Dropout(0.5),
                                            nn.Linear(prompt_dim * 2, prompt_dim * 4), nn.ReLU(), nn.Dropout(0.5),
                                            nn.Linear(prompt_dim * 4, prompt_dim * 4), nn.ReLU(), nn.Dropout(0.5),
                                            nn.Linear(prompt_dim * 4, prompt_dim), nn.ReLU(), nn.Dropout(0.5))
                self.meta_net1 = nn.Sequential(nn.Linear(prompt_dim, prompt_dim * 2), nn.ReLU(), nn.Dropout(0.5),
                                            nn.Linear(prompt_dim * 2, prompt_dim * 4), nn.ReLU(), nn.Dropout(0.5),
                                            nn.Linear(prompt_dim * 4, prompt_dim * 4), nn.ReLU(), nn.Dropout(0.5),
                                            nn.Linear(prompt_dim * 4, prompt_dim), nn.ReLU(), nn.Dropout(0.5))
                self.meta_net2 = nn.Sequential(nn.Linear(prompt_dim, prompt_dim * 2), nn.ReLU(), nn.Dropout(0.5),
                                            nn.Linear(prompt_dim * 2, prompt_dim * 4), nn.ReLU(), nn.Dropout(0.5),
                                            nn.Linear(prompt_dim * 4, prompt_dim * 4), nn.ReLU(), nn.Dropout(0.5),
                                            nn.Linear(prompt_dim * 4, prompt_dim), nn.ReLU(), nn.Dropout(0.5))
            # 20240315
            elif args.date == '20240315':
                """ unshared meta net weight"""
                self.meta_nets = []
                for _ in range(args.n_ctx):
                    meta_net = nn.Sequential(nn.Linear(prompt_dim, prompt_dim * 2), nn.ReLU(), nn.Dropout(0.5),
                                            nn.Linear(prompt_dim * 2, prompt_dim * 4), nn.ReLU(), nn.Dropout(0.5),
                                            nn.Linear(prompt_dim * 4, prompt_dim * 4), nn.ReLU(), nn.Dropout(0.5),
                                            nn.Linear(prompt_dim * 4, prompt_dim), nn.ReLU(), nn.Dropout(0.5))
                    meta_net.cuda(args.gpu)      
                    self.meta_nets.append(meta_net)
                    
        self.prompt_vectors = nn.Parameter(torch.empty(args.n_ctx, prompt_dim))
        nn.init.normal_(self.prompt_vectors, std=0.02)
        
        prompt = " ".join(["X"] * args.n_ctx)
        text = prompt + " "
        text_token = tokenize(text)
        text_token = text_token.cuda(args.gpu, non_blocking=True)
        text_token = text_token.view(1, -1)
        
        with torch.no_grad():
            text_embedding = clip_model.token_embedding(text_token).type(dtype)
         
        self.text_token = text_token
        self.register_buffer("embedding_prefix", text_embedding[:, :1, :]) # SOS
        self.register_buffer("embedding_suffix", text_embedding[:, 1+args.n_ctx: 2+args.n_ctx, :]) # EOS, CLS
          
        self.num_k = num_k
        self.topk = topk
        self.epsilon = epsilon
        self.phi_s = phi_s
        self.phi_a = phi_a
        self.temperature = temperature

        self.local_atte_fc = nn.Sequential(nn.Linear(img_patch_dim, token_feat), nn.Sigmoid())

        self.cosine_criterion = nn.CosineEmbeddingLoss()
        self.criterion_target = torch.as_tensor([1])

        self.transformer = transformer
        self.templates = nn.Parameter(torch.randn(1, num_k, img_patch_dim))

    def get_latent_local_attributes_feats(self, featuremap):
        batch_size = featuremap.shape[0]
        feature_dim = featuremap.shape[2]

        initial_templates = self.templates.expand(batch_size, self.num_k, feature_dim)
        cat_feature = torch.cat([initial_templates, featuremap], dim=1)
        latent_local_feats = self.transformer(cat_feature, mask=None)[:, :self.num_k, :]
        latent_local_feats = self.local_atte_fc(latent_local_feats)

        return latent_local_feats

    def get_img_local_attr_feats(self, img_global_feat, image_patch_feats):
        bs = image_patch_feats.shape[0]  # [128, 257, 1024]
        latent_local_feats = self.get_latent_local_attributes_feats(image_patch_feats)

        # Preliminary screening based on attention score
        attention_weights = torch.matmul(latent_local_feats, img_global_feat.unsqueeze(dim=2)).squeeze(dim=2)
        attention_weights = F.softmax(attention_weights, dim=1)

        local_attr_num = []
        sorted_indices = torch.argsort(attention_weights, dim=1, descending=True)
        sorted_indices = sorted_indices[:, :self.topk]
        selected_local_feats = []

        for i in range(bs):
            mask = attention_weights[i] > self.epsilon
            non_indices = torch.nonzero(mask).squeeze()
            num_r = non_indices.numel() if non_indices.numel() < self.topk else self.topk
            if num_r < 1:
                num_r = 1
            # Ensure the order of attribute features
            select_indices = sorted_indices[i][:num_r]
            select_indices = torch.sort(select_indices, dim=0).values
            select_id = torch.cat((select_indices, sorted_indices[i][num_r:]), dim=0)
            local_attr_num.append(num_r)
            selected_local_feats.append(latent_local_feats[i, select_id, :])

        selected_local_feats = torch.stack(selected_local_feats, dim=0)

        return F.normalize(selected_local_feats, dim=-1), local_attr_num

    def img_to_text(self, img, clip_model, modification_text):
        # inference
        with torch.no_grad():
            # Extract global and patch features from the image
            img_global_feat = clip_model.encode_image(img)
            img_patch_feats = get_img_patch_feats(img, clip_model)

            # Get local attribute features and count
            img_local_attr_feat, local_attr_num = self.get_img_local_attr_feats(img_global_feat, img_patch_feats)
            img_subj_token = self.phi_s(img_global_feat)
            img_attr_tokens = self.phi_a(img_local_attr_feat)

            text_list = []
            bs = img_global_feat.shape[0]
            for i in range(bs):
                # Generate the composed description for each image
                text = f" * with {'* ' * local_attr_num[i]}but " + modification_text[i]
                text_list.append(text)

            # Tokenize the composed description
            text = clip.tokenize(text_list, truncate=True).cuda(non_blocking=True)
            # Replace tokens to obtain pseudo-word-based features
            rest_pseudo_word_based_feat, collect_ind = prompt_token_replace(local_attr_num, text, img_subj_token, img_attr_tokens, clip_model, 0)
            text_embedding = self.prompt_learner(img_global_feat, rest_pseudo_word_based_feat)
            pseudo_word_based_feat = encode_text_img_learnable(text_embedding, collect_ind, clip_model)
            
        return pseudo_word_based_feat

    def cosine_loss(self, pseudo_word_based_feat, img_global_feat):
        cosine_loss = self.cosine_criterion(img_global_feat, pseudo_word_based_feat, self.criterion_target.cuda())
        return cosine_loss

    def get_templates(self, local_attr_num):
        template_list = []
        bs = len(local_attr_num)
        for i in range(bs):
            template = f" * with {'* ' * local_attr_num[i]}"
            template_list.append(template)
        templates = clip.tokenize(template_list, truncate=True).cuda(non_blocking=True)
        return templates

    def getLoss(self, images, subject, attribute, clip_model):
        with torch.no_grad():
            img_global_feat = clip_model.encode_image(images)  # [batch_size, 768]
            img_patch_feats = get_img_patch_feats(images, clip_model)  # [batch_size, channel_dim, feature_dim]
            
        # Obtain local fine-grained features
        img_salient_local_feats, local_attr_num = self.get_img_local_attr_feats(img_global_feat, img_patch_feats)
        # Perform token mapping
        img_subj_token = self.phi_s(img_global_feat)
        img_attr_tokens = self.phi_a(img_salient_local_feats)

        templates = self.get_templates(local_attr_num)
        rest_pseudo_word_based_feat, collect_ind = prompt_token_replace(local_attr_num, templates, img_subj_token, img_attr_tokens, clip_model, 0)
        text_embedding = self.prompt_learner(img_global_feat, rest_pseudo_word_based_feat)
        pseudo_word_based_feat = encode_text_img_learnable(text_embedding, collect_ind, clip_model)

        # compute the total loss
        # img_text_loss = contrastive_loss2(img_global_feat, pseudo_word_based_feat, clip_model)
        img_text_loss = contrastive_loss(img_global_feat, pseudo_word_based_feat, self.temperature)
       
        return img_text_loss
    
    def prompt_learner(self, token_features, rest_features):
        prefix = self.embedding_prefix
        suffix = self.embedding_suffix
        if self.args.meta_prompt:
            # 20240224
            if self.args.date == '20240224':
                bias = self.meta_net(token_features) # [B, 768]
                bias = bias.unsqueeze(1) # [B, 1, 768]
                ctx = self.prompt_vectors.unsqueeze(0) # [1, 3, 768]
                context = ctx + bias
            # 20240227
            elif self.args.date == '20240227':
                bias0 = self.meta_net0(token_features).unsqueeze(1) # (B, 1, 768)
                bias1 = self.meta_net1(token_features).unsqueeze(1) # (B, 1, 768)
                bias2 = self.meta_net2(token_features).unsqueeze(1) # (B, 1, 768)
                ctx0 = self.prompt_vectors[0].unsqueeze(0)
                ctx1 = self.prompt_vectors[1].unsqueeze(0)
                ctx2 = self.prompt_vectors[2].unsqueeze(0) # [1, 3, 768]
                ctx_shifted0 = ctx0 + bias0
                ctx_shifted1 = ctx1 + bias1
                ctx_shifted2 = ctx2 + bias2
                context = torch.cat([ctx_shifted0, ctx_shifted1, ctx_shifted2], dim=1)
            # 20240830
            elif self.args.date == '20240830':
                ctx0 = self.prompt_vectors[0].unsqueeze(0)
                ctx1 = self.prompt_vectors[1].unsqueeze(0)
                ctx2 = self.prompt_vectors[2].unsqueeze(0)
                ctx_shifted0 = ctx0
                ctx_shifted1 = ctx1
                ctx_shifted2 = ctx2
                context = torch.cat([ctx_shifted0, ctx_shifted1, ctx_shifted2], dim=0)
            # 20240315
            elif self.args.date == '20240315':
                bias = []
                ctx_shifteds = []
                for i in range(self.args.n_ctx):
                    b = self.meta_nets[i](token_features).unsqueeze(1) # (B, 1, 768)
                    ctx = self.prompt_vectors[i].unsqueeze(0) # (1, 1, 768)
                    ctx_shifted = ctx + b
                    ctx_shifteds.append(ctx_shifted)
                context = torch.cat(ctx_shifteds, dim=1) # (B, 3, 768)
            # pdb.set_trace()
            text_embedding = self.construct_prompts(prefix, suffix, context, rest_features)
            # pdb.set_trace()
        else:
            text_embedding = self.construct_prompts(prefix, suffix, self.prompt_vectors, rest_features)
        # pdb.set_trace()
        return text_embedding
    
    def construct_prompts(self, prefix, suffix, prompt_vectors, rest_features):
        bsz = rest_features.shape[0]
        # n_ctx = prompt_vectors.shape[1]
        if rest_features.dim() == 2:
            rest_features = rest_features.reshape(bsz, 1, -1)
        if prompt_vectors.dim() == 2:
            prompt_vectors = prompt_vectors.unsqueeze(0).expand(bsz, -1, -1)
        
        # pdb.set_trace()
        text_embedding = torch.cat(
            [
                prefix.expand(bsz, -1, -1),
                prompt_vectors,
                rest_features[:, self.args.n_ctx + 1: -1, :],
                suffix.expand(bsz, -1, -1)
            ],
            dim=1
        )
        return text_embedding