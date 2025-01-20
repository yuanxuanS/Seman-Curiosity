from .encoder import GraphAttentionEncoder
from .decoder import Decoder

class Atten_Policy(NNBase):
    def __init__(self, input_shape, num_actions,
                 recurrent=True, hidden_size=512,
                 num_sem_categories=5, **kwargs):
        super(Atten_Policy, self).__init__(
            recurrent, hidden_size, hidden_size)
        
        num_heads = kwargs['num_heads'] if 'num_heads' in kwargs else 8
        embedding_dim = kwargs['embedding_dim'] if 'embedding_dim' in kwargs else 128
        num_encoder_layers = kwargs['num_encoder_layers'] if 'num_encoder_layers' in kwargs else 3
        normalization = kwargs['normalization'] if 'normalization' in kwargs else "batch"
        
        self.encoder = GraphAttentionEncoder(
                num_heads=num_heads,
                embedding_dim=embedding_dim,
                num_layers=num_encoder_layers,
                normalization=normalization,
            )
        self.decoder = Decoder(
                env_name=self.env_name,
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                use_graph_context=use_graph_context,    # TODO
                mask_inner=mask_inner,  # TODO
                context_embedding=context_embedding,    # TODO
                dynamic_embedding=dynamic_embedding,    # TODO
            )
        self.critic_linear = nn.Linear(hidden_size // 2, 1)
        
        self.train()
        
    def forward(self, rgb, rnn_hxs, masks):
        embeddings = self.encoder()
        x = self.decoder()
        
        # 辅助loss
        self.feature = x
        
        return self.critic_linear(x).squeeze(-1), x, rnn_hxs
    
    
    def loss(self, target):
        pred_loc = self.pred_linear(self.feature)
        return entropy(pred_loc, target)
    
    
        